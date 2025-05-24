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
      .value("NONE", libyuv::kFilterNone)
      .value("Linear", libyuv::kFilterLinear)
      .value("Bilinear", libyuv::kFilterBilinear)
      .value("Box", libyuv::kFilterBox);

  nb::enum_<libyuv::FourCC>(m, "FourCC")
      .value("I420", libyuv::FOURCC_I420)
      .value("I422", libyuv::FOURCC_I422)
      .value("I444", libyuv::FOURCC_I444)
      .value("I400", libyuv::FOURCC_I400)
      .value("NV21", libyuv::FOURCC_NV21)
      .value("NV12", libyuv::FOURCC_NV12)
      .value("YUY2", libyuv::FOURCC_YUY2)
      .value("UYVY", libyuv::FOURCC_UYVY)
      .value("I010", libyuv::FOURCC_I010)
      .value("I210", libyuv::FOURCC_I210)
      .value("M420", libyuv::FOURCC_M420)
      .value("ARGB", libyuv::FOURCC_ARGB)
      .value("BGRA", libyuv::FOURCC_BGRA)
      .value("ABGR", libyuv::FOURCC_ABGR)
      .value("AR30", libyuv::FOURCC_AR30)
      .value("AB30", libyuv::FOURCC_AB30)
      .value("AR64", libyuv::FOURCC_AR64)
      .value("AB64", libyuv::FOURCC_AB64)
      .value("24BG", libyuv::FOURCC_24BG)
      .value("RAW", libyuv::FOURCC_RAW)
      .value("RGBA", libyuv::FOURCC_RGBA)
      .value("RGBP", libyuv::FOURCC_RGBP)
      .value("RGBO", libyuv::FOURCC_RGBO)
      .value("R444", libyuv::FOURCC_R444)
      .value("MJPG", libyuv::FOURCC_MJPG)
      .value("YV12", libyuv::FOURCC_YV12)
      .value("YV16", libyuv::FOURCC_YV16)
      .value("YV24", libyuv::FOURCC_YV24)
      .value("YU12", libyuv::FOURCC_YU12)
      .value("J420", libyuv::FOURCC_J420)
      .value("J422", libyuv::FOURCC_J422)
      .value("J444", libyuv::FOURCC_J444)
      .value("J400", libyuv::FOURCC_J400)
      .value("F420", libyuv::FOURCC_F420)
      .value("F422", libyuv::FOURCC_F422)
      .value("F444", libyuv::FOURCC_F444)
      .value("H420", libyuv::FOURCC_H420)
      .value("H422", libyuv::FOURCC_H422)
      .value("H444", libyuv::FOURCC_H444)
      .value("U420", libyuv::FOURCC_U420)
      .value("U422", libyuv::FOURCC_U422)
      .value("U444", libyuv::FOURCC_U444)
      .value("F010", libyuv::FOURCC_F010)
      .value("H010", libyuv::FOURCC_H010)
      .value("U010", libyuv::FOURCC_U010)
      .value("F210", libyuv::FOURCC_F210)
      .value("H210", libyuv::FOURCC_H210)
      .value("U210", libyuv::FOURCC_U210)
      .value("P010", libyuv::FOURCC_P010)
      .value("P210", libyuv::FOURCC_P210)
      .value("IYUV", libyuv::FOURCC_IYUV)
      .value("YU16", libyuv::FOURCC_YU16)
      .value("YU24", libyuv::FOURCC_YU24)
      .value("YUYV", libyuv::FOURCC_YUYV)
      .value("YUVS", libyuv::FOURCC_YUVS)
      .value("HDYC", libyuv::FOURCC_HDYC)
      .value("2VUY", libyuv::FOURCC_2VUY)
      .value("JPEG", libyuv::FOURCC_JPEG)
      .value("DMB1", libyuv::FOURCC_DMB1)
      .value("BA81", libyuv::FOURCC_BA81)
      .value("RGB3", libyuv::FOURCC_RGB3)
      .value("BGR3", libyuv::FOURCC_BGR3)
      .value("CM32", libyuv::FOURCC_CM32)
      .value("CM24", libyuv::FOURCC_CM24)
      .value("L555", libyuv::FOURCC_L555)
      .value("L565", libyuv::FOURCC_L565)
      .value("5551", libyuv::FOURCC_5551)
      .value("I411", libyuv::FOURCC_I411)
      .value("Q420", libyuv::FOURCC_Q420)
      .value("RGGB", libyuv::FOURCC_RGGB)
      .value("BGGR", libyuv::FOURCC_BGGR)
      .value("GRBG", libyuv::FOURCC_GRBG)
      .value("GBRG", libyuv::FOURCC_GBRG)
      .value("H264", libyuv::FOURCC_H264)
      .value("ANY", libyuv::FOURCC_ANY)
      .export_values();

  nb::enum_<libyuv::RotationMode>(m, "RotationMode")
      .value("Rotate0", libyuv::kRotate0)
      .value("Rotate90", libyuv::kRotate90)
      .value("Rotate180", libyuv::kRotate180)
      .value("Rotate270", libyuv::kRotate270)
      .export_values();

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
