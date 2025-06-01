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

class AomVideoDecoder : public VideoDecoder {
 public:
  AomVideoDecoder();
  ~AomVideoDecoder();

  // VideoDecoder interface
  bool Init(const Settings& settings) override;
  void Release() override;
  void Decode(const EncodedImage& encoded_image) override;
  void SetOnDecode(std::function<void(const VideoFrame&)> on_decode) override;

 private:
  aom_codec_ctx_t codec_ = {};
  bool initialized_ = false;
  std::function<void(const VideoFrame&)> on_decode_;
};

AomVideoDecoder::AomVideoDecoder() {
}

AomVideoDecoder::~AomVideoDecoder() {
  Release();
}

bool AomVideoDecoder::Init(const Settings& settings) {
  if (settings.codec_type != VideoCodecType::AV1) {
    PLOG_ERROR << "AomVideoDecoder: Unsupported codec type";
    return false;
  }

  Release();

  aom_codec_dec_cfg_t cfg = {};
  cfg.threads = 1;  // シングルスレッドで開始
  cfg.allow_lowbitdepth = 1;

  aom_codec_err_t res = aom_codec_dec_init(&codec_, aom_codec_av1_dx(), &cfg, 0);
  if (res != AOM_CODEC_OK) {
    PLOG_ERROR << "AomVideoDecoder: Failed to initialize decoder: " 
               << aom_codec_err_to_string(res);
    return false;
  }

  initialized_ = true;
  return true;
}

void AomVideoDecoder::Release() {
  if (initialized_) {
    aom_codec_destroy(&codec_);
    initialized_ = false;
  }
}

void AomVideoDecoder::Decode(const EncodedImage& encoded_image) {
  if (!initialized_ || !on_decode_) {
    PLOG_ERROR << "AomVideoDecoder: decoder not initialized or on_decode_ is null";
    return;
  }

  PLOG_INFO << "AomVideoDecoder: Decoding " << encoded_image.data.size() << " bytes";

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
      PLOG_WARNING << "AomVideoDecoder: Unsupported image format: " << img->fmt;
      continue;
    }

    int width = img->d_w;
    int height = img->d_h;

    auto i420_buffer = VideoFrameBufferI420::Create(width, height);

    // Y平面のコピー
    uint8_t* src_y = img->planes[AOM_PLANE_Y];
    uint8_t* dst_y = (uint8_t*)i420_buffer->y.data();
    int y_stride = img->stride[AOM_PLANE_Y];
    for (int i = 0; i < height; i++) {
      memcpy(dst_y + i * width, src_y + i * y_stride, width);
    }

    // U平面のコピー
    uint8_t* src_u = img->planes[AOM_PLANE_U];
    uint8_t* dst_u = (uint8_t*)i420_buffer->u.data();
    int u_stride = img->stride[AOM_PLANE_U];
    for (int i = 0; i < height / 2; i++) {
      memcpy(dst_u + i * width / 2, src_u + i * u_stride, width / 2);
    }

    // V平面のコピー
    uint8_t* src_v = img->planes[AOM_PLANE_V];
    uint8_t* dst_v = (uint8_t*)i420_buffer->v.data();
    int v_stride = img->stride[AOM_PLANE_V];
    for (int i = 0; i < height / 2; i++) {
      memcpy(dst_v + i * width / 2, src_v + i * v_stride, width / 2);
    }

    VideoFrame frame;
    frame.format = ImageFormat::I420;
    frame.i420_buffer = i420_buffer;
    frame.timestamp = encoded_image.timestamp;
    frame.rid = encoded_image.rid;
    frame.base_width = width;
    frame.base_height = height;

    on_decode_(frame);
  }
}

void AomVideoDecoder::SetOnDecode(
    std::function<void(const VideoFrame&)> on_decode) {
  on_decode_ = on_decode;
}

std::shared_ptr<VideoDecoder> CreateAomVideoDecoder() {
  return std::make_shared<AomVideoDecoder>();
}