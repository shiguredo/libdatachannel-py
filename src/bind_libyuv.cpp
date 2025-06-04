// nanobind
#include <nanobind/make_iterator.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/chrono.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>
#include <nanobind/trampoline.h>

#include <libyuv.h>

namespace nb = nanobind;
using namespace nb::literals;

extern void bind_libyuv(nb::module_& m) {
  nb::enum_<libyuv::FilterMode>(m, "FilterMode")
      .value("kNone", libyuv::kFilterNone)
      .value("kLinear", libyuv::kFilterLinear)
      .value("kBilinear", libyuv::kFilterBilinear)
      .value("kBox", libyuv::kFilterBox);

  nb::enum_<libyuv::FourCC>(m, "FourCC")
      .value("kI420", libyuv::FOURCC_I420)
      .value("kI422", libyuv::FOURCC_I422)
      .value("kI444", libyuv::FOURCC_I444)
      .value("kI400", libyuv::FOURCC_I400)
      .value("kNV21", libyuv::FOURCC_NV21)
      .value("kNV12", libyuv::FOURCC_NV12)
      .value("kYUY2", libyuv::FOURCC_YUY2)
      .value("kUYVY", libyuv::FOURCC_UYVY)
      .value("kI010", libyuv::FOURCC_I010)
      .value("kI210", libyuv::FOURCC_I210)
      .value("kM420", libyuv::FOURCC_M420)
      .value("kARGB", libyuv::FOURCC_ARGB)
      .value("kBGRA", libyuv::FOURCC_BGRA)
      .value("kABGR", libyuv::FOURCC_ABGR)
      .value("kAR30", libyuv::FOURCC_AR30)
      .value("kAB30", libyuv::FOURCC_AB30)
      .value("kAR64", libyuv::FOURCC_AR64)
      .value("kAB64", libyuv::FOURCC_AB64)
      .value("k24BG", libyuv::FOURCC_24BG)
      .value("kRAW", libyuv::FOURCC_RAW)
      .value("kRGBA", libyuv::FOURCC_RGBA)
      .value("kRGBP", libyuv::FOURCC_RGBP)
      .value("kRGBO", libyuv::FOURCC_RGBO)
      .value("kR444", libyuv::FOURCC_R444)
      .value("kMJPG", libyuv::FOURCC_MJPG)
      .value("kYV12", libyuv::FOURCC_YV12)
      .value("kYV16", libyuv::FOURCC_YV16)
      .value("kYV24", libyuv::FOURCC_YV24)
      .value("kYU12", libyuv::FOURCC_YU12)
      .value("kJ420", libyuv::FOURCC_J420)
      .value("kJ422", libyuv::FOURCC_J422)
      .value("kJ444", libyuv::FOURCC_J444)
      .value("kJ400", libyuv::FOURCC_J400)
      .value("kF420", libyuv::FOURCC_F420)
      .value("kF422", libyuv::FOURCC_F422)
      .value("kF444", libyuv::FOURCC_F444)
      .value("kH420", libyuv::FOURCC_H420)
      .value("kH422", libyuv::FOURCC_H422)
      .value("kH444", libyuv::FOURCC_H444)
      .value("kU420", libyuv::FOURCC_U420)
      .value("kU422", libyuv::FOURCC_U422)
      .value("kU444", libyuv::FOURCC_U444)
      .value("kF010", libyuv::FOURCC_F010)
      .value("kH010", libyuv::FOURCC_H010)
      .value("kU010", libyuv::FOURCC_U010)
      .value("kF210", libyuv::FOURCC_F210)
      .value("kH210", libyuv::FOURCC_H210)
      .value("kU210", libyuv::FOURCC_U210)
      .value("kP010", libyuv::FOURCC_P010)
      .value("kP210", libyuv::FOURCC_P210)
      .value("kIYUV", libyuv::FOURCC_IYUV)
      .value("kYU16", libyuv::FOURCC_YU16)
      .value("kYU24", libyuv::FOURCC_YU24)
      .value("kYUYV", libyuv::FOURCC_YUYV)
      .value("kYUVS", libyuv::FOURCC_YUVS)
      .value("kHDYC", libyuv::FOURCC_HDYC)
      .value("k2VUY", libyuv::FOURCC_2VUY)
      .value("kJPEG", libyuv::FOURCC_JPEG)
      .value("kDMB1", libyuv::FOURCC_DMB1)
      .value("kBA81", libyuv::FOURCC_BA81)
      .value("kRGB3", libyuv::FOURCC_RGB3)
      .value("kBGR3", libyuv::FOURCC_BGR3)
      .value("kCM32", libyuv::FOURCC_CM32)
      .value("kCM24", libyuv::FOURCC_CM24)
      .value("kL555", libyuv::FOURCC_L555)
      .value("kL565", libyuv::FOURCC_L565)
      .value("k5551", libyuv::FOURCC_5551)
      .value("kI411", libyuv::FOURCC_I411)
      .value("kQ420", libyuv::FOURCC_Q420)
      .value("kRGGB", libyuv::FOURCC_RGGB)
      .value("kBGGR", libyuv::FOURCC_BGGR)
      .value("kGRBG", libyuv::FOURCC_GRBG)
      .value("kGBRG", libyuv::FOURCC_GBRG)
      .value("kH264", libyuv::FOURCC_H264)
      .value("kANY", libyuv::FOURCC_ANY);

  nb::enum_<libyuv::RotationMode>(m, "RotationMode")
      .value("kRotate0", libyuv::kRotate0)
      .value("kRotate90", libyuv::kRotate90)
      .value("kRotate180", libyuv::kRotate180)
      .value("kRotate270", libyuv::kRotate270);

  m.def(
      "nv12_scale",
      [](const nb::ndarray<>& src_y, const nb::ndarray<>& src_uv,
         int src_stride_y, int src_stride_uv, int src_width, int src_height,
         nb::ndarray<>& dst_y, nb::ndarray<>& dst_uv, int dst_stride_y,
         int dst_stride_uv, int dst_width, int dst_height,
         libyuv::FilterMode filtering) {
        return libyuv::NV12Scale(
            static_cast<const uint8_t*>(src_y.data()), src_stride_y,
            static_cast<const uint8_t*>(src_uv.data()), src_stride_uv,
            src_width, src_height, static_cast<uint8_t*>(dst_y.data()),
            dst_stride_y, static_cast<uint8_t*>(dst_uv.data()), dst_stride_uv,
            dst_width, dst_height, filtering);
      },
      "src_y"_a, "src_uv"_a, "src_stride_y"_a, "src_stride_uv"_a, "src_width"_a,
      "src_height"_a, "dst_y"_a, "dst_uv"_a, "dst_stride_y"_a,
      "dst_stride_uv"_a, "dst_width"_a, "dst_height"_a, "filtering"_a);

  m.def(
      "i420_scale",
      [](const nb::ndarray<>& src_y, const nb::ndarray<>& src_u,
         const nb::ndarray<>& src_v, int src_stride_y, int src_stride_u,
         int src_stride_v, int src_width, int src_height, nb::ndarray<>& dst_y,
         nb::ndarray<>& dst_u, nb::ndarray<>& dst_v, int dst_stride_y,
         int dst_stride_u, int dst_stride_v, int dst_width, int dst_height,
         libyuv::FilterMode filtering) {
        return libyuv::I420Scale(
            static_cast<const uint8_t*>(src_y.data()), src_stride_y,
            static_cast<const uint8_t*>(src_u.data()), src_stride_u,
            static_cast<const uint8_t*>(src_v.data()), src_stride_v, src_width,
            src_height, static_cast<uint8_t*>(dst_y.data()), dst_stride_y,
            static_cast<uint8_t*>(dst_u.data()), dst_stride_u,
            static_cast<uint8_t*>(dst_v.data()), dst_stride_v, dst_width,
            dst_height, filtering);
      },
      "src_y"_a, "src_u"_a, "src_v"_a, "src_stride_y"_a, "src_stride_u"_a,
      "src_stride_v"_a, "src_width"_a, "src_height"_a, "dst_y"_a, "dst_u"_a,
      "dst_v"_a, "dst_stride_y"_a, "dst_stride_u"_a, "dst_stride_v"_a,
      "dst_width"_a, "dst_height"_a, "filtering"_a);

  m.def(
      "convert_to_i420",
      [](const nb::ndarray<>& sample, size_t sample_size, nb::ndarray<>& dst_y,
         int dst_stride_y, nb::ndarray<>& dst_u, int dst_stride_u,
         nb::ndarray<>& dst_v, int dst_stride_v, int crop_x, int crop_y,
         int src_width, int src_height, int crop_width, int crop_height,
         libyuv::RotationMode rotation, libyuv::FourCC fourcc) {
        return libyuv::ConvertToI420(
            static_cast<const uint8_t*>(sample.data()), sample_size,
            static_cast<uint8_t*>(dst_y.data()), dst_stride_y,
            static_cast<uint8_t*>(dst_u.data()), dst_stride_u,
            static_cast<uint8_t*>(dst_v.data()), dst_stride_v, crop_x, crop_y,
            src_width, src_height, crop_width, crop_height, rotation, fourcc);
      },
      "sample"_a, "sample_size"_a, "dst_y"_a, "dst_stride_y"_a, "dst_u"_a,
      "dst_stride_u"_a, "dst_v"_a, "dst_stride_v"_a, "crop_x"_a, "crop_y"_a,
      "src_width"_a, "src_height"_a, "crop_width"_a, "crop_height"_a,
      "rotation"_a, "fourcc"_a);

  // NV12 <-> I420
  m.def(
      "nv12_to_i420",
      [](const nb::ndarray<>& src_y, const nb::ndarray<>& src_uv,
         int src_stride_y, int src_stride_uv, nb::ndarray<>& dst_y,
         nb::ndarray<>& dst_u, nb::ndarray<>& dst_v, int dst_stride_y,
         int dst_stride_u, int dst_stride_v, int width, int height) {
        return libyuv::NV12ToI420(
            static_cast<const uint8_t*>(src_y.data()), src_stride_y,
            static_cast<const uint8_t*>(src_uv.data()), src_stride_uv,
            static_cast<uint8_t*>(dst_y.data()), dst_stride_y,
            static_cast<uint8_t*>(dst_u.data()), dst_stride_u,
            static_cast<uint8_t*>(dst_v.data()), dst_stride_v, width, height);
      },
      "src_y"_a, "src_uv"_a, "src_stride_y"_a, "src_stride_uv"_a, "dst_y"_a,
      "dst_u"_a, "dst_v"_a, "dst_stride_y"_a, "dst_stride_u"_a,
      "dst_stride_v"_a, "width"_a, "height"_a);

  m.def(
      "i420_to_nv12",
      [](const nb::ndarray<>& src_y, const nb::ndarray<>& src_u,
         const nb::ndarray<>& src_v, int src_stride_y, int src_stride_u,
         int src_stride_v, nb::ndarray<>& dst_y, nb::ndarray<>& dst_uv,
         int dst_stride_y, int dst_stride_uv, int width, int height) {
        return libyuv::I420ToNV12(
            static_cast<const uint8_t*>(src_y.data()), src_stride_y,
            static_cast<const uint8_t*>(src_u.data()), src_stride_u,
            static_cast<const uint8_t*>(src_v.data()), src_stride_v,
            static_cast<uint8_t*>(dst_y.data()), dst_stride_y,
            static_cast<uint8_t*>(dst_uv.data()), dst_stride_uv, width, height);
      },
      "src_y"_a, "src_u"_a, "src_v"_a, "src_stride_y"_a, "src_stride_u"_a,
      "src_stride_v"_a, "dst_y"_a, "dst_uv"_a, "dst_stride_y"_a,
      "dst_stride_uv"_a, "width"_a, "height"_a);

  // RGB24 <-> I420
  m.def(
      "rgb24_to_i420",
      [](const nb::ndarray<>& src_rgb24, int src_stride_rgb24,
         nb::ndarray<>& dst_y, nb::ndarray<>& dst_u, nb::ndarray<>& dst_v,
         int dst_stride_y, int dst_stride_u, int dst_stride_v, int width,
         int height) {
        return libyuv::RGB24ToI420(
            static_cast<const uint8_t*>(src_rgb24.data()), src_stride_rgb24,
            static_cast<uint8_t*>(dst_y.data()), dst_stride_y,
            static_cast<uint8_t*>(dst_u.data()), dst_stride_u,
            static_cast<uint8_t*>(dst_v.data()), dst_stride_v, width, height);
      },
      "src_rgb24"_a, "src_stride_rgb24"_a, "dst_y"_a, "dst_u"_a, "dst_v"_a,
      "dst_stride_y"_a, "dst_stride_u"_a, "dst_stride_v"_a, "width"_a,
      "height"_a);

  m.def(
      "i420_to_rgb24",
      [](const nb::ndarray<>& src_y, const nb::ndarray<>& src_u,
         const nb::ndarray<>& src_v, int src_stride_y, int src_stride_u,
         int src_stride_v, nb::ndarray<>& dst_rgb24, int dst_stride_rgb24,
         int width, int height) {
        return libyuv::I420ToRGB24(
            static_cast<const uint8_t*>(src_y.data()), src_stride_y,
            static_cast<const uint8_t*>(src_u.data()), src_stride_u,
            static_cast<const uint8_t*>(src_v.data()), src_stride_v,
            static_cast<uint8_t*>(dst_rgb24.data()), dst_stride_rgb24, width,
            height);
      },
      "src_y"_a, "src_u"_a, "src_v"_a, "src_stride_y"_a, "src_stride_u"_a,
      "src_stride_v"_a, "dst_rgb24"_a, "dst_stride_rgb24"_a, "width"_a,
      "height"_a);
}
