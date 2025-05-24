#include "video_codec.h"

video_buffer_type CreateVideoBuffer(int size) {
  auto ptr = new uint8_t[size]();
  nanobind::capsule owner(
      ptr, [](void* ptr) noexcept { delete[] static_cast<uint8_t*>(ptr); });
  return video_buffer_type(ptr, {(size_t)size}, owner, {(int64_t)1});
}

image_type CreateImage(int width, int height, int stride) {
  auto ptr = new uint8_t[stride * height]();
  nanobind::capsule owner(
      ptr, [](void* ptr) noexcept { delete[] static_cast<uint8_t*>(ptr); });
  return image_type(ptr, {(size_t)height, (size_t)width}, owner,
                    {(int64_t)stride, 1});
}

template <int N>
image_type_n<N> CreateImageN(int width, int height, int stride) {
  auto ptr = new uint8_t[stride * height]();
  nanobind::capsule owner(
      ptr, [](void* ptr) noexcept { delete[] static_cast<uint8_t*>(ptr); });
  return image_type_n<N>(ptr, {(size_t)height, (size_t)width, (size_t)N}, owner,
                         {(int64_t)stride, (int64_t)N, 1});
}

// VideoFrameBufferI420

int VideoFrameBufferI420::width() const {
  return y.shape(1);
}
int VideoFrameBufferI420::height() const {
  return y.shape(0);
}
int VideoFrameBufferI420::stride_y() const {
  return y.stride(0);
}
int VideoFrameBufferI420::stride_u() const {
  return u.stride(0);
}
int VideoFrameBufferI420::stride_v() const {
  return v.stride(0);
}

std::shared_ptr<VideoFrameBufferI420> VideoFrameBufferI420::Create(int width,
                                                                   int height) {
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

// VideoFrameBufferNV12

int VideoFrameBufferNV12::width() const {
  return y.shape(1);
}
int VideoFrameBufferNV12::height() const {
  return y.shape(0);
}
int VideoFrameBufferNV12::stride_y() const {
  return y.stride(0);
}
int VideoFrameBufferNV12::stride_uv() const {
  return uv.stride(0);
}

std::shared_ptr<VideoFrameBufferNV12> VideoFrameBufferNV12::Create(int width,
                                                                   int height) {
  auto p = std::make_shared<VideoFrameBufferNV12>();
  int stride_y = width;
  int stride_uv = (width + 1) / 2 * 2;
  int chroma_height = (height + 1) / 2;
  p->y = CreateImage(stride_y, height, stride_y);
  p->uv = CreateImage(stride_uv, chroma_height, stride_uv);
  return p;
}

// VideoFrameBufferBGR888

int VideoFrameBufferBGR888::width() const {
  return bgr.shape(1);
}
int VideoFrameBufferBGR888::height() const {
  return bgr.shape(0);
}
int VideoFrameBufferBGR888::stride() const {
  return bgr.stride(0);
}

std::shared_ptr<VideoFrameBufferBGR888> VideoFrameBufferBGR888::Create(
    int width,
    int height) {
  auto p = std::make_shared<VideoFrameBufferBGR888>();
  int stride = width * 3;
  p->bgr = CreateImageN<3>(width, height, stride);
  return p;
}

// VideoFrame

int VideoFrame::width() const {
  return format == ImageFormat::I420   ? i420_buffer->width()
         : format == ImageFormat::NV12 ? nv12_buffer->width()
                                       : bgr888_buffer->width();
}
int VideoFrame::height() const {
  return format == ImageFormat::I420   ? i420_buffer->height()
         : format == ImageFormat::NV12 ? nv12_buffer->height()
                                       : bgr888_buffer->height();
}
