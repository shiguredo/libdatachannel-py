#include "openh264_video_decoder.h"

#include <dlfcn.h>
#include <stdint.h>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>

// plog
#include <plog/Log.h>

// OpenH264
#include <wels/codec_api.h>
#include <wels/codec_app_def.h>
#include <wels/codec_def.h>

class OpenH264VideoDecoder : public VideoDecoder {
 public:
  OpenH264VideoDecoder(const std::string& openh264);
  ~OpenH264VideoDecoder();

  // VideoDecoder interface
  bool Init(const Settings& settings) override;
  void Release() override;
  void Decode(const EncodedImage& encoded_image) override;
  void SetOnDecode(std::function<void(const VideoFrame&)> on_decode) override;

 private:
  bool InitOpenH264(const std::string& openh264);
  void ReleaseOpenH264();

  void* openh264_handle_ = nullptr;
  ISVCDecoder* decoder_ = nullptr;
  std::function<void(const VideoFrame&)> on_decode_;

  // Function pointers
  using CreateDecoderFunc = int (*)(ISVCDecoder**);
  using DestroyDecoderFunc = void (*)(ISVCDecoder*);
  CreateDecoderFunc create_decoder_ = nullptr;
  DestroyDecoderFunc destroy_decoder_ = nullptr;
};

OpenH264VideoDecoder::OpenH264VideoDecoder(const std::string& openh264) {
  bool result = InitOpenH264(openh264);
  if (!result) {
    throw std::runtime_error("Failed to load OpenH264");
  }
}

OpenH264VideoDecoder::~OpenH264VideoDecoder() {
  Release();
  ReleaseOpenH264();
}

bool OpenH264VideoDecoder::InitOpenH264(const std::string& openh264) {
  void* handle = dlopen(openh264.c_str(), RTLD_LAZY);
  if (handle == nullptr) {
    PLOG_ERROR << "Failed to open OpenH264 library: " << dlerror();
    return false;
  }
  
  create_decoder_ = (CreateDecoderFunc)dlsym(handle, "WelsCreateDecoder");
  if (create_decoder_ == nullptr) {
    dlclose(handle);
    return false;
  }
  
  destroy_decoder_ = (DestroyDecoderFunc)dlsym(handle, "WelsDestroyDecoder");
  if (destroy_decoder_ == nullptr) {
    dlclose(handle);
    return false;
  }
  
  openh264_handle_ = handle;
  return true;
}

void OpenH264VideoDecoder::ReleaseOpenH264() {
  if (openh264_handle_ != nullptr) {
    dlclose(openh264_handle_);
    openh264_handle_ = nullptr;
  }
}

bool OpenH264VideoDecoder::Init(const Settings& settings) {
  if (settings.codec_type != VideoCodecType::H264) {
    PLOG_ERROR << "OpenH264VideoDecoder: Unsupported codec type";
    return false;
  }

  Release();

  int rv = create_decoder_(&decoder_);
  if (rv != 0 || decoder_ == nullptr) {
    PLOG_ERROR << "OpenH264VideoDecoder: Failed to create decoder";
    return false;
  }

  SDecodingParam decoding_param = {0};
  decoding_param.uiTargetDqLayer = UCHAR_MAX;
  decoding_param.eEcActiveIdc = ERROR_CON_FRAME_COPY;
  decoding_param.sVideoProperty.eVideoBsType = VIDEO_BITSTREAM_DEFAULT;

  rv = decoder_->Initialize(&decoding_param);
  if (rv != 0) {
    PLOG_ERROR << "OpenH264VideoDecoder: Failed to initialize decoder";
    destroy_decoder_(decoder_);
    decoder_ = nullptr;
    return false;
  }

  return true;
}

void OpenH264VideoDecoder::Release() {
  if (decoder_ != nullptr) {
    decoder_->Uninitialize();
    destroy_decoder_(decoder_);
    decoder_ = nullptr;
  }
}

void OpenH264VideoDecoder::Decode(const EncodedImage& encoded_image) {
  if (decoder_ == nullptr || !on_decode_) {
    PLOG_ERROR << "OpenH264VideoDecoder: decoder_ or on_decode_ is null";
    return;
  }

  uint8_t* dst[3] = {nullptr, nullptr, nullptr};
  SBufferInfo dst_info = {0};

  PLOG_INFO << "OpenH264VideoDecoder: Decoding " << encoded_image.data.size() << " bytes";

  DECODING_STATE rv = decoder_->DecodeFrame2(
      encoded_image.data.data(), encoded_image.data.size(), dst, &dst_info);

  PLOG_INFO << "OpenH264VideoDecoder: DecodeFrame2 returned " << rv << ", iBufferStatus=" << dst_info.iBufferStatus;

  if (rv != dsErrorFree && rv != dsNoParamSets) {
    PLOG_WARNING << "OpenH264VideoDecoder: Decode error: " << rv;
    return;
  }

  // バッファステータスが0の場合は、デコードが完了していない（パラメータセットのみ等）
  if (dst_info.iBufferStatus == 0) {
    PLOG_INFO << "OpenH264VideoDecoder: No frame output yet (buffering)";
    return;
  }

  if (dst_info.iBufferStatus == 1) {
    // デコード成功
    int width = dst_info.UsrData.sSystemBuffer.iWidth;
    int height = dst_info.UsrData.sSystemBuffer.iHeight;
    int y_stride = dst_info.UsrData.sSystemBuffer.iStride[0];
    int uv_stride = dst_info.UsrData.sSystemBuffer.iStride[1];

    auto i420_buffer = VideoFrameBufferI420::Create(width, height);

    // Y平面のコピー
    uint8_t* src_y = dst[0];
    uint8_t* dst_y = (uint8_t*)i420_buffer->y.data();
    for (int i = 0; i < height; i++) {
      memcpy(dst_y + i * width, src_y + i * y_stride, width);
    }

    // U平面のコピー
    uint8_t* src_u = dst[1];
    uint8_t* dst_u = (uint8_t*)i420_buffer->u.data();
    for (int i = 0; i < height / 2; i++) {
      memcpy(dst_u + i * width / 2, src_u + i * uv_stride, width / 2);
    }

    // V平面のコピー
    uint8_t* src_v = dst[2];
    uint8_t* dst_v = (uint8_t*)i420_buffer->v.data();
    for (int i = 0; i < height / 2; i++) {
      memcpy(dst_v + i * width / 2, src_v + i * uv_stride, width / 2);
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

void OpenH264VideoDecoder::SetOnDecode(
    std::function<void(const VideoFrame&)> on_decode) {
  on_decode_ = on_decode;
}

std::shared_ptr<VideoDecoder> CreateOpenH264VideoDecoder(
    const std::string& openh264) {
  return std::make_shared<OpenH264VideoDecoder>(openh264);
}