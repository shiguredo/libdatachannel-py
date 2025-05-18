#ifndef LIBDATACHANNEL_VIDEO_CODEC_H_INCLUDED
#define LIBDATACHANNEL_VIDEO_CODEC_H_INCLUDED

#include <stdint.h>
#include <chrono>
#include <functional>
#include <memory>
#include <optional>
#include <string>

// nanobind
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

using buffer_type = nanobind::
    ndarray<nanobind::numpy, uint8_t, nanobind::shape<-1>, nanobind::c_contig>;

using image_type = nanobind::ndarray<nanobind::numpy,
                                     uint8_t,
                                     nanobind::shape<-1, -1>,
                                     nanobind::c_contig>;
template <int N>
using image_type_n = nanobind::ndarray<nanobind::numpy,
                                       uint8_t,
                                       nanobind::shape<-1, -1, N>,
                                       nanobind::c_contig>;

namespace {

static buffer_type CreateBuffer(int size) {
  auto ptr = new uint8_t[size]();
  nanobind::capsule owner(
      ptr, [](void* ptr) noexcept { delete[] static_cast<uint8_t*>(ptr); });
  return buffer_type(ptr, {(size_t)size}, owner, {(int64_t)1});
}

static image_type CreateImage(int width, int height, int stride) {
  auto ptr = new uint8_t[stride * height]();
  nanobind::capsule owner(
      ptr, [](void* ptr) noexcept { delete[] static_cast<uint8_t*>(ptr); });
  return image_type(ptr, {(size_t)height, (size_t)width}, owner,
                    {(int64_t)stride, 1});
}

template <int N>
static image_type_n<N> CreateImageN(int width, int height, int stride) {
  auto ptr = new uint8_t[stride * height]();
  nanobind::capsule owner(
      ptr, [](void* ptr) noexcept { delete[] static_cast<uint8_t*>(ptr); });
  return image_type_n<N>(ptr, {(size_t)height, (size_t)width, (size_t)N}, owner,
                         {(int64_t)stride, (int64_t)N, 1});
}

}  // namespace

enum class VideoCodecType {
  H264,
  H265,
  VP8,
  VP9,
  AV1,
};

enum class ImageFormat {
  I420,
  NV12,
  BGR888,
};

struct VideoFrameBufferI420 {
  image_type y;
  image_type u;
  image_type v;
  int width() const { return y.shape(1); }
  int height() const { return y.shape(0); }
  int stride_y() const { return y.stride(0); }
  int stride_u() const { return u.stride(0); }
  int stride_v() const { return v.stride(0); }

  static std::shared_ptr<VideoFrameBufferI420> Create(int width, int height) {
    auto p = std::make_shared<VideoFrameBufferI420>();
    int stride_y = width;
    int stride_u = (width + 1) / 2;
    int stride_v = (width + 1) / 2;
    int chroma_height = (height + 1) / 2;
    p->y = CreateImage(stride_y, height, stride_y);
    p->u = CreateImage(stride_u, chroma_height, stride_u);
    p->v = CreateImage(stride_v, chroma_height, stride_v);
    return p;
  }
};

struct VideoFrameBufferNV12 {
  image_type y;
  image_type uv;
  int width() const { return y.shape(1); }
  int height() const { return y.shape(0); }
  int stride_y() const { return y.stride(0); }
  int stride_uv() const { return uv.stride(0); }

  static std::shared_ptr<VideoFrameBufferNV12> Create(int width, int height) {
    auto p = std::make_shared<VideoFrameBufferNV12>();
    int stride_y = width;
    int stride_uv = (width + 1) / 2 * 2;
    int chroma_height = (height + 1) / 2;
    p->y = CreateImage(stride_y, height, stride_y);
    p->uv = CreateImage(stride_uv, chroma_height, stride_uv);
    return p;
  }
};

struct VideoFrameBufferBGR888 {
  image_type_n<3> bgr;
  int width() const { return bgr.shape(1); }
  int height() const { return bgr.shape(0); }
  int stride() const { return bgr.stride(0); }

  static std::shared_ptr<VideoFrameBufferBGR888> Create(int width, int height) {
    auto p = std::make_shared<VideoFrameBufferBGR888>();
    int stride = width * 3;
    p->bgr = CreateImageN<3>(width, height, stride);
    return p;
  }
};

struct VideoFrame {
  ImageFormat format;
  std::shared_ptr<VideoFrameBufferI420> i420_buffer;
  std::shared_ptr<VideoFrameBufferNV12> nv12_buffer;
  std::shared_ptr<VideoFrameBufferBGR888> bgr888_buffer;
  std::chrono::microseconds timestamp;
  std::optional<std::string> rid;
  int base_width;
  int base_height;
  int width() const {
    return format == ImageFormat::I420   ? i420_buffer->width()
           : format == ImageFormat::NV12 ? nv12_buffer->width()
                                         : bgr888_buffer->width();
  }
  int height() const {
    return format == ImageFormat::I420   ? i420_buffer->height()
           : format == ImageFormat::NV12 ? nv12_buffer->height()
                                         : bgr888_buffer->height();
  }
};

struct EncodedImage {
  buffer_type data;
  std::chrono::microseconds timestamp;
  std::optional<std::string> rid;
};

class VideoEncoder {
 public:
  struct Settings {
    VideoCodecType codec_type;
    int width;
    int height;
    size_t bitrate;
  };

  virtual ~VideoEncoder() = default;

  virtual bool Init(const Settings& settings) = 0;
  virtual void Release() = 0;
  virtual void Encode(const VideoFrame& frame) = 0;
  virtual void ForceIntraNextFrame() = 0;
  virtual void SetOnEncoded(
      std::function<void(const EncodedImage&)> on_encoded) = 0;
};

#endif
