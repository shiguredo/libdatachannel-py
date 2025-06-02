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

#include "aom_video_decoder.h"
#include "aom_video_encoder.h"
#include "audio_codec.h"
#include "openh264_video_decoder.h"
#include "openh264_video_encoder.h"
#include "opus_audio_decoder.h"
#include "opus_audio_encoder.h"
#include "video_codec.h"
#include "videotoolbox_video_encoder.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

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
      .def_rw("frame_number", &VideoFrame::frame_number)
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
      .def("set_on_encode", &VideoEncoder::SetOnEncode);

  // VideoEncoder::Settings
  nb::class_<VideoEncoder::Settings>(encoder, "Settings")
      .def(nb::init<>())
      .def_rw("codec_type", &VideoEncoder::Settings::codec_type)
      .def_rw("width", &VideoEncoder::Settings::width)
      .def_rw("height", &VideoEncoder::Settings::height)
      .def_rw("bitrate", &VideoEncoder::Settings::bitrate)
      .def_rw("fps", &VideoEncoder::Settings::fps);

  m.def("create_openh264_video_encoder", &CreateOpenH264VideoEncoder,
        "openh264"_a);
  m.def("create_videotoolbox_video_encoder", &CreateVideoToolboxVideoEncoder);
  m.def("create_aom_video_encoder", &CreateAOMVideoEncoder);

  // VideoDecoder
  nb::class_<VideoDecoder> decoder(m, "VideoDecoder");
  decoder.def("init", &VideoDecoder::Init)
      .def("release", &VideoDecoder::Release)
      .def("decode", &VideoDecoder::Decode)
      .def("set_on_decode", &VideoDecoder::SetOnDecode);

  // VideoDecoder::Settings
  nb::class_<VideoDecoder::Settings>(decoder, "Settings")
      .def(nb::init<>())
      .def_rw("codec_type", &VideoDecoder::Settings::codec_type);

  m.def("create_openh264_video_decoder", &CreateOpenH264VideoDecoder,
        "openh264"_a);
  m.def("create_aom_video_decoder", &CreateAomVideoDecoder);
}

void bind_audio_codec(nb::module_& m) {
  nb::enum_<AudioCodecType>(m, "AudioCodecType")
      .value("OPUS", AudioCodecType::OPUS);

  nb::class_<AudioFrame>(m, "AudioFrame")
      .def(nb::init<>())
      .def_rw("sample_rate", &AudioFrame::sample_rate)
      .def_rw("pcm", &AudioFrame::pcm, nb::rv_policy::reference)
      .def_rw("timestamp", &AudioFrame::timestamp)
      .def("channels", &AudioFrame::channels)
      .def("samples", &AudioFrame::samples);

  nb::class_<EncodedAudio>(m, "EncodedAudio")
      .def(nb::init<>())
      .def_rw("data", &EncodedAudio::data, nb::rv_policy::reference)
      .def_rw("timestamp", &EncodedAudio::timestamp);

  // AudioEncoder
  nb::class_<AudioEncoder> encoder(m, "AudioEncoder");
  encoder.def("init", &AudioEncoder::Init)
      .def("release", &AudioEncoder::Release)
      .def("encode", &AudioEncoder::Encode)
      .def("set_on_encode", &AudioEncoder::SetOnEncode);

  // AudioEncoder::Settings
  nb::class_<AudioEncoder::Settings>(encoder, "Settings")
      .def(nb::init<>())
      .def_rw("codec_type", &AudioEncoder::Settings::codec_type)
      .def_rw("sample_rate", &AudioEncoder::Settings::sample_rate)
      .def_rw("channels", &AudioEncoder::Settings::channels)
      .def_rw("bitrate", &AudioEncoder::Settings::bitrate)
      .def_rw("frame_duration_ms", &AudioEncoder::Settings::frame_duration_ms)
      .def_rw("opus_inband_fec", &AudioEncoder::Settings::opus_inband_fec);

  m.def("create_opus_audio_encoder", &CreateOpusAudioEncoder);

  // AudioDecoder
  nb::class_<AudioDecoder> decoder(m, "AudioDecoder");
  decoder.def("init", &AudioDecoder::Init)
      .def("release", &AudioDecoder::Release)
      .def("decode", &AudioDecoder::Decode)
      .def("set_on_decode", &AudioDecoder::SetOnDecode);

  // AudioDecoder::Settings
  nb::class_<AudioDecoder::Settings>(decoder, "Settings")
      .def(nb::init<>())
      .def_rw("codec_type", &AudioDecoder::Settings::codec_type)
      .def_rw("sample_rate", &AudioDecoder::Settings::sample_rate)
      .def_rw("channels", &AudioDecoder::Settings::channels);

  m.def("create_opus_audio_decoder", &CreateOpusAudioDecoder);
}

}  // namespace

void bind_codec(nb::module_& m) {
  bind_video_codec(m);
  bind_audio_codec(m);
}
