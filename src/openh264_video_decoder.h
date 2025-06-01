#ifndef LIBDATACHANNEL_OPENH264_VIDEO_DECODER_H_INCLUDED
#define LIBDATACHANNEL_OPENH264_VIDEO_DECODER_H_INCLUDED

#include <memory>
#include <string>

#include "video_codec.h"

std::shared_ptr<VideoDecoder> CreateOpenH264VideoDecoder(
    const std::string& openh264);

#endif