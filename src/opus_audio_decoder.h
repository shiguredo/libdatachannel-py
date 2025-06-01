#ifndef LIBDATACHANNEL_OPUS_AUDIO_DECODER_H_INCLUDED
#define LIBDATACHANNEL_OPUS_AUDIO_DECODER_H_INCLUDED

#include "audio_codec.h"

std::shared_ptr<AudioDecoder> CreateOpusAudioDecoder();

#endif