#include "aom_video_encoder.h"

#include <string.h>
#include <atomic>
#include <exception>

// plog
#include <plog/Log.h>

// libdatachannel
// #include <rtc/rtppacketizationconfig.hpp>

// AOM
#include <aom/aom_codec.h>
#include <aom/aom_encoder.h>
#include <aom/aomcx.h>

// text の定義を全て展開した上で文字列化する。
// 単純に #text とした場合、全て展開する前に文字列化されてしまう
#if defined(_WIN32)
#define STRINGIZE(text) STRINGIZE_((text))
#define STRINGIZE_(x) STRINGIZE_I x
#else
#define STRINGIZE(x) STRINGIZE_I(x)
#endif

#define STRINGIZE_I(text) #text

class AOMVideoEncoder : public VideoEncoder {
 public:
  AOMVideoEncoder() = default;
  ~AOMVideoEncoder() override { Release(); }

  void ForceIntraNextFrame() override { next_iframe_ = true; }

  bool Init(const Settings& settings) override {
    Release();

    PLOG_INFO << "AOM InitEncode";

    settings_ = settings;

    // https://source.chromium.org/chromium/chromium/src/+/main:third_party/webrtc/modules/video_coding/codecs/av1/libaom_av1_encoder.cc
    // を参考に初期化やエンコードを行う

    aom_codec_err_t ret = aom_codec_enc_config_default(
        aom_codec_av1_cx(), &cfg_, AOM_USAGE_REALTIME);
    if (ret != AOM_CODEC_OK) {
      PLOG_ERROR << "Failed to aom_codec_enc_config_default: ret=" << ret;
      return false;
    }

    // Overwrite default config with input encoder settings & RTC-relevant values.
    cfg_.g_w = settings.width;
    cfg_.g_h = settings.height;
    cfg_.g_threads = 8;
    cfg_.g_timebase.num = 1;
    cfg_.g_timebase.den = 90000;
    cfg_.rc_target_bitrate = settings.bitrate / 1000;  // kbps
    cfg_.rc_dropframe_thresh = 0;
    cfg_.g_input_bit_depth = 8;
    cfg_.kf_mode = AOM_KF_DISABLED;
    cfg_.rc_min_quantizer = 10;
    cfg_.rc_max_quantizer = 63;
    cfg_.rc_undershoot_pct = 50;
    cfg_.rc_overshoot_pct = 50;
    cfg_.rc_buf_initial_sz = 600;
    cfg_.rc_buf_optimal_sz = 600;
    cfg_.rc_buf_sz = 1000;
    cfg_.g_usage = AOM_USAGE_REALTIME;
    cfg_.g_error_resilient = 0;
    // Low-latency settings.
    cfg_.rc_end_usage = AOM_CBR;    // Constant Bit Rate (CBR) mode
    cfg_.g_pass = AOM_RC_ONE_PASS;  // One-pass rate control
    cfg_.g_lag_in_frames = 0;       // No look ahead when lag equals 0.

    if (frame_for_encode_ != nullptr) {
      aom_img_free(frame_for_encode_);
      frame_for_encode_ = nullptr;
    }

    // Flag options: AOM_CODEC_USE_PSNR and AOM_CODEC_USE_HIGHBITDEPTH
    aom_codec_flags_t flags = 0;

    // Initialize an encoder instance.
    ret = aom_codec_enc_init_ver(&ctx_, aom_codec_av1_cx(), &cfg_, flags,
                                 AOM_ENCODER_ABI_VERSION);
    if (ret != AOM_CODEC_OK) {
      PLOG_ERROR << "Failed to aom_codec_enc_init_ver: ret=" << ret;
      return false;
    }
    init_ctx_ = true;

#define SET_PARAM(param_id, param_value)                                    \
  do {                                                                      \
    ret = aom_codec_control(&ctx_, param_id, param_value);                  \
    if (ret != AOM_CODEC_OK) {                                              \
      PLOG_ERROR << "Failed to aom_codec_control: ret=" << ret              \
                 << ", param_id="                                           \
                 << STRINGIZE(param_id) << ", param_value=" << param_value; \
      return false;                                                         \
    }                                                                       \
  } while (0)

    // Set control parameters
    SET_PARAM(AOME_SET_CPUUSED, 10);
    SET_PARAM(AV1E_SET_ENABLE_CDEF, 1);
    SET_PARAM(AV1E_SET_ENABLE_TPL_MODEL, 0);
    SET_PARAM(AV1E_SET_DELTAQ_MODE, 0);
    SET_PARAM(AV1E_SET_ENABLE_ORDER_HINT, 0);
    SET_PARAM(AV1E_SET_AQ_MODE, 3);
    SET_PARAM(AOME_SET_MAX_INTRA_BITRATE_PCT, 300);
    SET_PARAM(AV1E_SET_COEFF_COST_UPD_FREQ, 3);
    SET_PARAM(AV1E_SET_MODE_COST_UPD_FREQ, 3);
    SET_PARAM(AV1E_SET_MV_COST_UPD_FREQ, 3);

    SET_PARAM(AV1E_SET_ENABLE_PALETTE, 0);

    SET_PARAM(AV1E_SET_TILE_ROWS, 1);
    SET_PARAM(AV1E_SET_TILE_COLUMNS, 2);

    SET_PARAM(AV1E_SET_ROW_MT, 1);
    SET_PARAM(AV1E_SET_ENABLE_OBMC, 0);
    SET_PARAM(AV1E_SET_NOISE_SENSITIVITY, 0);
    SET_PARAM(AV1E_SET_ENABLE_WARPED_MOTION, 0);
    SET_PARAM(AV1E_SET_ENABLE_GLOBAL_MOTION, 0);
    SET_PARAM(AV1E_SET_ENABLE_REF_FRAME_MVS, 0);
    SET_PARAM(AV1E_SET_SUPERBLOCK_SIZE, AOM_SUPERBLOCK_SIZE_DYNAMIC);
    SET_PARAM(AV1E_SET_ENABLE_CFL_INTRA, 0);
    SET_PARAM(AV1E_SET_ENABLE_SMOOTH_INTRA, 0);
    SET_PARAM(AV1E_SET_ENABLE_ANGLE_DELTA, 0);
    SET_PARAM(AV1E_SET_ENABLE_FILTER_INTRA, 0);
    SET_PARAM(AV1E_SET_INTRA_DEFAULT_TX_ONLY, 1);
    SET_PARAM(AV1E_SET_DISABLE_TRELLIS_QUANT, 1);
    SET_PARAM(AV1E_SET_ENABLE_DIST_WTD_COMP, 0);
    SET_PARAM(AV1E_SET_ENABLE_DIFF_WTD_COMP, 0);
    SET_PARAM(AV1E_SET_ENABLE_DUAL_FILTER, 0);
    SET_PARAM(AV1E_SET_ENABLE_INTERINTRA_COMP, 0);
    SET_PARAM(AV1E_SET_ENABLE_INTERINTRA_WEDGE, 0);
    SET_PARAM(AV1E_SET_ENABLE_INTRA_EDGE_FILTER, 0);
    SET_PARAM(AV1E_SET_ENABLE_INTRABC, 0);
    SET_PARAM(AV1E_SET_ENABLE_MASKED_COMP, 0);
    SET_PARAM(AV1E_SET_ENABLE_PAETH_INTRA, 0);
    SET_PARAM(AV1E_SET_ENABLE_QM, 0);
    SET_PARAM(AV1E_SET_ENABLE_RECT_PARTITIONS, 0);
    SET_PARAM(AV1E_SET_ENABLE_RESTORATION, 0);
    SET_PARAM(AV1E_SET_ENABLE_SMOOTH_INTERINTRA, 0);
    SET_PARAM(AV1E_SET_ENABLE_TX64, 0);
    SET_PARAM(AV1E_SET_MAX_REFERENCE_FRAMES, 3);

    return true;
  }

  void SetOnEncode(
      std::function<void(const EncodedImage&)> on_encode) override {
    on_encode_ = on_encode;
  }

  void Encode(const VideoFrame& frame) override {
    if (frame.format != ImageFormat::I420 &&
        frame.format != ImageFormat::NV12) {
      PLOG_ERROR << "Unknown video frame format";
      return;
    }
    aom_img_fmt_t fmt =
        frame.format == ImageFormat::I420 ? AOM_IMG_FMT_I420 : AOM_IMG_FMT_NV12;

    if (frame_for_encode_ == nullptr || frame_for_encode_->fmt != fmt) {
      if (frame_for_encode_ != nullptr) {
        aom_img_free(frame_for_encode_);
      }
      frame_for_encode_ =
          aom_img_wrap(nullptr, fmt, cfg_.g_w, cfg_.g_h, 1, nullptr);
    }

    if (frame.format == ImageFormat::I420) {
      // I420
      frame_for_encode_->planes[AOM_PLANE_Y] = frame.i420_buffer->y.data();
      frame_for_encode_->planes[AOM_PLANE_U] = frame.i420_buffer->u.data();
      frame_for_encode_->planes[AOM_PLANE_V] = frame.i420_buffer->v.data();
      frame_for_encode_->stride[AOM_PLANE_Y] = frame.i420_buffer->stride_y();
      frame_for_encode_->stride[AOM_PLANE_U] = frame.i420_buffer->stride_u();
      frame_for_encode_->stride[AOM_PLANE_V] = frame.i420_buffer->stride_v();
    } else {
      // NV12
      frame_for_encode_->planes[AOM_PLANE_Y] = frame.nv12_buffer->y.data();
      frame_for_encode_->planes[AOM_PLANE_U] = frame.nv12_buffer->uv.data();
      frame_for_encode_->planes[AOM_PLANE_V] = nullptr;
      frame_for_encode_->stride[AOM_PLANE_Y] = frame.nv12_buffer->stride_y();
      frame_for_encode_->stride[AOM_PLANE_U] = frame.nv12_buffer->stride_uv();
      frame_for_encode_->stride[AOM_PLANE_V] = 0;
    }

    const uint32_t duration = 90000 / settings_.fps;
    timestamp_ += duration;

    aom_enc_frame_flags_t flags = 0;

    bool send_key_frame = next_iframe_.exchange(false);
    if (send_key_frame) {
      PLOG_DEBUG << "KeyFrame generated";
      flags = AOM_EFLAG_FORCE_KF;
    }

    aom_codec_err_t ret =
        aom_codec_encode(&ctx_, frame_for_encode_, timestamp_, duration, flags);

    EncodedImage encoded;
    const aom_codec_cx_pkt_t* pkt = nullptr;
    aom_codec_iter_t iter = nullptr;
    while (true) {
      const aom_codec_cx_pkt_t* p = aom_codec_get_cx_data(&ctx_, &iter);
      if (p == nullptr) {
        break;
      }
      if (p->kind == AOM_CODEC_CX_FRAME_PKT && p->data.frame.sz > 0) {
        pkt = p;
      }
    }

    encoded.data = CreateVideoBuffer(pkt->data.frame.sz);
    memcpy(encoded.data.data(), pkt->data.frame.buf, encoded.data.size());
    encoded.timestamp = frame.timestamp;

    bool is_key_frame = (pkt->data.frame.flags & AOM_EFLAG_FORCE_KF) != 0;

    // // DD の設定を行う
    // rtc::DependencyDescriptorContext ctx;
    // ctx.structure.templateIdOffset = 0;
    // ctx.structure.decodeTargetCount = 1;
    // ctx.structure.chainCount = 1;
    // ctx.structure.decodeTargetProtectedBy = {0};
    // ctx.structure.resolutions.push_back({frame.width(), frame.height()});
    // rtc::FrameDependencyTemplate key_frame_template;
    // key_frame_template.spatialId = 0;
    // key_frame_template.temporalId = 0;
    // key_frame_template.decodeTargetIndications = {
    //     rtc::DecodeTargetIndication::Switch};
    // key_frame_template.chainDiffs = {0};
    // rtc::FrameDependencyTemplate delta_frame_template;
    // delta_frame_template.spatialId = 0;
    // delta_frame_template.temporalId = 0;
    // delta_frame_template.decodeTargetIndications = {
    //     rtc::DecodeTargetIndication::Switch};
    // delta_frame_template.chainDiffs = {1};
    // delta_frame_template.frameDiffs = {1};
    // ctx.structure.templates = {key_frame_template, delta_frame_template};
    // ctx.activeChains[0] = true;
    // ctx.descriptor.frameNumber = frame.frame_number;
    // if (is_key_frame) {
    //   ctx.descriptor.dependencyTemplate = key_frame_template;
    // } else {
    //   ctx.descriptor.dependencyTemplate = delta_frame_template;
    //   ctx.descriptor.dependencyTemplate.frameDiffs = {frame.frame_number -
    //                                                   prev_frame_number_};
    // }
    // ctx.descriptor.structureAttached = is_key_frame;

    // encoded.dependency_descriptor_context =
    //     std::make_shared<rtc::DependencyDescriptorContext>(ctx);

    prev_frame_number_ = frame.frame_number;

    on_encode_(encoded);
  }

  void Release() override {
    if (frame_for_encode_ != nullptr) {
      aom_img_free(frame_for_encode_);
      frame_for_encode_ = nullptr;
    }
    if (init_ctx_) {
      aom_codec_destroy(&ctx_);
      init_ctx_ = false;
    }
  }

 private:
  Settings settings_;
  bool init_ctx_ = false;
  aom_codec_ctx_t ctx_;
  aom_codec_enc_cfg_t cfg_;
  aom_image_t* frame_for_encode_ = nullptr;
  int64_t timestamp_ = 0;
  int prev_frame_number_ = 0;

  std::function<void(const EncodedImage&)> on_encode_;

  std::atomic<bool> next_iframe_;
};

std::shared_ptr<VideoEncoder> CreateAOMVideoEncoder() {
  return std::make_shared<AOMVideoEncoder>();
}
