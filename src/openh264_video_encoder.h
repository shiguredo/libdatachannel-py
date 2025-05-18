#ifndef LIBDATACHANNEL_OPENH264_VIDEO_ENCODER_H_INCLUDED
#define LIBDATACHANNEL_OPENH264_VIDEO_ENCODER_H_INCLUDED

#include "video_codec.h"

std::shared_ptr<VideoEncoder> CreateOpenH264VideoEncoder(std::string openh264);

#endif
