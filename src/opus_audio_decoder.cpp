#include "opus_audio_decoder.h"

#include <chrono>
#include <functional>
#include <vector>

// opus
#include <opus.h>

// plog
#include <plog/Log.h>

class OpusAudioDecoder : public AudioDecoder {
 public:
  OpusAudioDecoder() {}
  ~OpusAudioDecoder() override { Release(); }

  bool Init(const Settings& settings) override {
    Release();
    settings_ = settings;

    if (settings_.channels != 1 && settings_.channels != 2) {
      PLOG_ERROR << "Invalid channels: " << settings_.channels;
      return false;
    }

    int error = 0;
    decoder_ = opus_decoder_create(settings_.sample_rate, settings_.channels, &error);
    if (error != OPUS_OK) {
      PLOG_ERROR << "Failed to create opus decoder";
      return false;
    }

    // デコード用のバッファを初期化
    // Opus では最大 5760 サンプル (120ms @ 48kHz) まで
    pcm_buf_.resize(5760 * settings_.channels);
    return true;
  }

  void Release() override {
    if (decoder_ != nullptr) {
      opus_decoder_destroy(decoder_);
      decoder_ = nullptr;
    }
  }

  void Decode(const EncodedAudio& encoded) override {
    if (decoder_ == nullptr) {
      PLOG_ERROR << "Decoder not initialized";
      return;
    }

    int samples = opus_decode_float(decoder_, encoded.data.data(), encoded.data.shape(0),
                                    pcm_buf_.data(), pcm_buf_.size() / settings_.channels, 0);
    if (samples < 0) {
      PLOG_ERROR << "Failed to opus_decode_float: result=" << samples;
      return;
    }

    // デコードされたデータを AudioFrame に変換
    AudioFrame frame;
    frame.sample_rate = settings_.sample_rate;
    frame.timestamp = encoded.timestamp;
    frame.pcm = CreatePCMFloat(samples, settings_.channels);
    
    // PCMデータをコピー
    // frame.pcm.data() で生のポインタを取得
    float* pcm_data = static_cast<float*>(frame.pcm.data());
    memcpy(pcm_data, pcm_buf_.data(), samples * settings_.channels * sizeof(float));

    if (on_decode_) {
      on_decode_(frame);
    }
  }

  void SetOnDecode(
      std::function<void(const AudioFrame&)> on_decode) override {
    on_decode_ = on_decode;
  }

 private:
  Settings settings_;

  OpusDecoder* decoder_ = nullptr;
  std::function<void(const AudioFrame&)> on_decode_;
  std::vector<float> pcm_buf_;
};

std::shared_ptr<AudioDecoder> CreateOpusAudioDecoder() {
  return std::make_shared<OpusAudioDecoder>();
}