#ifndef LIBDATACHANNEL_AOM_VIDEO_DECODER_H_INCLUDED
#define LIBDATACHANNEL_AOM_VIDEO_DECODER_H_INCLUDED

#include <memory>
#include <string>

#include "video_codec.h"

std::shared_ptr<VideoDecoder> CreateAomVideoDecoder();

#endif