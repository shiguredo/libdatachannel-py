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

#include "openh264_video_encoder.h"
#include "video_codec.h"

namespace nb = nanobind;
using namespace nb::literals;

void bind_video_codec(nb::module_& m) {
  nb::enum_<VideoCodecType>(m, "VideoCodecType")
      .value("H264", VideoCodecType::H264)
      .value("H265", VideoCodecType::H265)
      .value("VP8", VideoCodecType::VP8)
      .value("VP9", VideoCodecType::VP9)
      .value("AV1", VideoCodecType::AV1);

  nb::enum_<ImageFormat>(m, "ImageFormat")
      .value("I420", ImageFormat::I420)
      .value("NV12", ImageFormat::NV12)
      .value("BGR888", ImageFormat::BGR888);

  nb::class_<VideoFrameBufferI420>(m, "VideoFrameBufferI420")
      .def_static("create", &VideoFrameBufferI420::Create)
      .def_rw("y", &VideoFrameBufferI420::y, nb::rv_policy::reference)
      .def_rw("u", &VideoFrameBufferI420::u, nb::rv_policy::reference)
      .def_rw("v", &VideoFrameBufferI420::v, nb::rv_policy::reference)
      .def("width", &VideoFrameBufferI420::width)
      .def("height", &VideoFrameBufferI420::height)
      .def("stride_y", &VideoFrameBufferI420::stride_y)
      .def("stride_u", &VideoFrameBufferI420::stride_u)
      .def("stride_v", &VideoFrameBufferI420::stride_v);

  nb::class_<VideoFrameBufferNV12>(m, "VideoFrameBufferNV12")
      .def_static("create", &VideoFrameBufferNV12::Create)
      .def_rw("y", &VideoFrameBufferNV12::y, nb::rv_policy::reference)
      .def_rw("uv", &VideoFrameBufferNV12::uv, nb::rv_policy::reference)
      .def("width", &VideoFrameBufferNV12::width)
      .def("height", &VideoFrameBufferNV12::height)
      .def("stride_y", &VideoFrameBufferNV12::stride_y)
      .def("stride_uv", &VideoFrameBufferNV12::stride_uv);

  nb::class_<VideoFrameBufferBGR888>(m, "VideoFrameBufferBGR888")
      .def_static("create", &VideoFrameBufferBGR888::Create)
      .def_rw("bgr", &VideoFrameBufferBGR888::bgr, nb::rv_policy::reference)
      .def("width", &VideoFrameBufferBGR888::width)
      .def("height", &VideoFrameBufferBGR888::height)
      .def("stride", &VideoFrameBufferBGR888::stride);

  // VideoFrame
  nb::class_<VideoFrame>(m, "VideoFrame")
      .def(nb::init<>())
      .def_rw("format", &VideoFrame::format)
      .def_rw("i420_buffer", &VideoFrame::i420_buffer)
      .def_rw("nv12_buffer", &VideoFrame::nv12_buffer)
      .def_rw("bgr888_buffer", &VideoFrame::bgr888_buffer)
      .def_rw("timestamp", &VideoFrame::timestamp)
      .def_rw("rid", &VideoFrame::rid)
      .def_rw("base_width", &VideoFrame::base_width)
      .def_rw("base_height", &VideoFrame::base_height)
      .def("width", &VideoFrame::width)
      .def("height", &VideoFrame::height);

  // EncodedImage
  nb::class_<EncodedImage>(m, "EncodedImage")
      .def(nb::init<>())
      .def_rw("data", &EncodedImage::data, nb::rv_policy::reference)
      .def_rw("timestamp", &EncodedImage::timestamp)
      .def_rw("rid", &EncodedImage::rid);

  // VideoEncoder
  nb::class_<VideoEncoder> encoder(m, "VideoEncoder");
  encoder.def("init", &VideoEncoder::Init)
      .def("release", &VideoEncoder::Release)
      .def("encode", &VideoEncoder::Encode)
      .def("force_intra_next_frame", &VideoEncoder::ForceIntraNextFrame)
      .def("set_on_encoded", &VideoEncoder::SetOnEncoded);

  // VideoEncoder::Settings
  nb::class_<VideoEncoder::Settings>(encoder, "Settings")
      .def(nb::init<>())
      .def_rw("codec_type", &VideoEncoder::Settings::codec_type)
      .def_rw("width", &VideoEncoder::Settings::width)
      .def_rw("height", &VideoEncoder::Settings::height)
      .def_rw("bitrate", &VideoEncoder::Settings::bitrate);

  m.def("create_openh264_video_encoder", &CreateOpenH264VideoEncoder,
        "openh264"_a);
}
