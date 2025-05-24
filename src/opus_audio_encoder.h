#ifndef LIBDATACHANNEL_OPUS_AUDIO_ENCODER_H_INCLUDED
#define LIBDATACHANNEL_OPUS_AUDIO_ENCODER_H_INCLUDED

#include "audio_codec.h"

std::shared_ptr<AudioEncoder> CreateOpusAudioEncoder();

#endif
