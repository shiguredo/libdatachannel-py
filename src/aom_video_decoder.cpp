#include "aom_video_decoder.h"

#include <stdint.h>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>

// plog
#include <plog/Log.h>

// libaom
#include <aom/aom_decoder.h>
#include <aom/aomdx.h>

// libyuv
#include <libyuv.h>

class AomVideoDecoder : public VideoDecoder {
 public:
  AomVideoDecoder() = default;
  ~AomVideoDecoder() { Release(); }

  // VideoDecoder interface
  bool Init(const Settings& settings) override {
    if (settings.codec_type != VideoCodecType::AV1) {
      PLOG_ERROR << "AomVideoDecoder: Unsupported codec type";
      return false;
    }

    Release();

    aom_codec_dec_cfg_t cfg = {};
    cfg.threads = 1;  // シングルスレッドで開始
    cfg.allow_lowbitdepth = 1;

    aom_codec_err_t res =
        aom_codec_dec_init(&codec_, aom_codec_av1_dx(), &cfg, 0);
    if (res != AOM_CODEC_OK) {
      PLOG_ERROR << "AomVideoDecoder: Failed to initialize decoder: "
                 << aom_codec_err_to_string(res);
      return false;
    }

    initialized_ = true;
    return true;
  }

  void Release() override {
    if (initialized_) {
      aom_codec_destroy(&codec_);
      initialized_ = false;
    }
  }

  void Decode(const EncodedImage& encoded_image) override {
    if (!initialized_) {
      PLOG_ERROR << "AomVideoDecoder: decoder not initialized";
      return;
    }

    PLOG_DEBUG << "AomVideoDecoder: Decoding " << encoded_image.data.size()
               << " bytes";

    aom_codec_err_t res = aom_codec_decode(&codec_, encoded_image.data.data(),
                                           encoded_image.data.size(), nullptr);

    if (res != AOM_CODEC_OK) {
      PLOG_WARNING << "AomVideoDecoder: Decode error: "
                   << aom_codec_err_to_string(res);
      return;
    }

    aom_codec_iter_t iter = nullptr;
    aom_image_t* img = nullptr;

    while ((img = aom_codec_get_frame(&codec_, &iter)) != nullptr) {
      if (img->fmt != AOM_IMG_FMT_I420 && img->fmt != AOM_IMG_FMT_I42016) {
        PLOG_WARNING << "AomVideoDecoder: Unsupported image format: "
                     << img->fmt;
        continue;
      }

      int width = img->d_w;
      int height = img->d_h;

      auto i420_buffer = VideoFrameBufferI420::Create(width, height);

      libyuv::I420Copy(img->planes[AOM_PLANE_Y], img->stride[AOM_PLANE_Y],
                       img->planes[AOM_PLANE_U], img->stride[AOM_PLANE_U],
                       img->planes[AOM_PLANE_V], img->stride[AOM_PLANE_V],
                       i420_buffer->y.data(), i420_buffer->stride_y(),
                       i420_buffer->u.data(), i420_buffer->stride_u(),
                       i420_buffer->v.data(), i420_buffer->stride_v(), width,
                       height);

      VideoFrame frame;
      frame.format = ImageFormat::I420;
      frame.i420_buffer = i420_buffer;
      frame.timestamp = encoded_image.timestamp;
      frame.rid = encoded_image.rid;
      frame.base_width = width;
      frame.base_height = height;

      if (on_decode_) {
        on_decode_(frame);
      }
    }
  }

  void SetOnDecode(std::function<void(const VideoFrame&)> on_decode) override {
    on_decode_ = on_decode;
  }

 private:
  aom_codec_ctx_t codec_ = {};
  bool initialized_ = false;
  std::function<void(const VideoFrame&)> on_decode_;
};

std::shared_ptr<VideoDecoder> CreateAomVideoDecoder() {
  return std::make_shared<AomVideoDecoder>();
}