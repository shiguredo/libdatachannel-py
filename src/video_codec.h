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

buffer_type CreateBuffer(int size);
image_type CreateImage(int width, int height, int stride);

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
  int width() const;
  int height() const;
  int stride_y() const;
  int stride_u() const;
  int stride_v() const;

  static std::shared_ptr<VideoFrameBufferI420> Create(int width, int height);
};

struct VideoFrameBufferNV12 {
  image_type y;
  image_type uv;
  int width() const;
  int height() const;
  int stride_y() const;
  int stride_uv() const;

  static std::shared_ptr<VideoFrameBufferNV12> Create(int width, int height);
};

struct VideoFrameBufferBGR888 {
  image_type_n<3> bgr;
  int width() const;
  int height() const;
  int stride() const;

  static std::shared_ptr<VideoFrameBufferBGR888> Create(int width, int height);
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
  int width() const;
  int height() const;
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
  virtual void SetOnEncode(
      std::function<void(const EncodedImage&)> on_encode) = 0;
};

#endif
