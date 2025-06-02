#ifndef LIBDATACHANNEL_AUDIO_CODEC_H_INCLUDED
#define LIBDATACHANNEL_AUDIO_CODEC_H_INCLUDED

#include <stdint.h>
#include <chrono>
#include <functional>
#include <memory>
#include <optional>
#include <string>

// nanobind
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

using audio_buffer_type = nanobind::
    ndarray<nanobind::numpy, uint8_t, nanobind::shape<-1>, nanobind::c_contig>;

using pcm_float_type = nanobind::ndarray<nanobind::numpy,
                                         float,
                                         nanobind::shape<-1, -1>,
                                         nanobind::c_contig>;

audio_buffer_type CreateAudioBuffer(int size);
pcm_float_type CreatePCMFloat(int samples, int channels);

enum class AudioCodecType {
  OPUS,
};

struct AudioFrame {
  int sample_rate = 0;
  pcm_float_type pcm;
  std::chrono::microseconds timestamp;
  int channels() const;
  int samples() const;
};

struct EncodedAudio {
  audio_buffer_type data;
  std::chrono::microseconds timestamp;
};

class AudioEncoder {
 public:
  struct Settings {
    AudioCodecType codec_type;
    int sample_rate;
    int channels;
    int bitrate;
    int frame_duration_ms;
    // 0 インバンドFECを無効にする（デフォルト）。
    // 1 インバンドFECを有効にする。パケットロス率が十分に高い場合、Opusは高ビットレートでも自動的にSILKに切り替えて、そのFECを使用できるようにする。
    // 2 インバンドFECを有効にするが、音楽がある場合でも必ずしもSILKに切り替えるとは限らない。
    int opus_inband_fec = 0;
  };

  virtual ~AudioEncoder() = default;

  virtual bool Init(const Settings& settings) = 0;
  virtual void Release() = 0;
  virtual void Encode(const AudioFrame& frame) = 0;
  virtual void SetOnEncode(
      std::function<void(const EncodedAudio&)> on_encode) = 0;
};

class AudioDecoder {
 public:
  struct Settings {
    AudioCodecType codec_type;
    int sample_rate;
    int channels;
  };

  virtual ~AudioDecoder() = default;

  virtual bool Init(const Settings& settings) = 0;
  virtual void Release() = 0;
  virtual void Decode(const EncodedAudio& encoded) = 0;
  virtual void SetOnDecode(
      std::function<void(const AudioFrame&)> on_decode) = 0;
};

#endif
