#include "opus_audio_encoder.h"

#include <chrono>
#include <functional>
#include <vector>

// opus
#include <opus.h>

// plog
#include <plog/Log.h>

class OpusAudioEncoder : public AudioEncoder {
 public:
  OpusAudioEncoder() {}
  ~OpusAudioEncoder() override { Release(); }

  bool Init(const Settings& settings) override {
    Release();
    settings_ = settings;

    if (settings_.channels != 1 && settings_.channels != 2) {
      PLOG_ERROR << "Invalid channels: " << settings_.channels;
      return false;
    }

    int error = 0;
    // maxplaybackrate=48000;stereo=1;sprop-stereo=1;minptime=10;ptime=20;useinbandfec=1;usedtx=0
    encoder_ = opus_encoder_create(settings_.sample_rate, settings_.channels,
                                   OPUS_APPLICATION_VOIP, &error);
    if (error != OPUS_OK) {
      PLOG_ERROR << "Failed to create opus encoder";
      return false;
    }
    int r = opus_encoder_ctl(encoder_, OPUS_SET_INBAND_FEC(1));
    if (r != OPUS_OK) {
      PLOG_ERROR << "Failed to OPUS_SET_INBAND_FEC";
      return false;
    }
    r = opus_encoder_ctl(encoder_, OPUS_SET_BITRATE(settings_.bitrate));
    if (r != OPUS_OK) {
      PLOG_ERROR << "Failed to OPUS_SET_BITRATE";
      return false;
    }

    encoded_buf_.resize(1024);
    return true;
  }
  void Release() override {
    if (encoder_ != nullptr) {
      opus_encoder_destroy(encoder_);
      encoder_ = nullptr;
    }
  }

  void Encode(const AudioFrame& frame) override {
    if (frame.sample_rate != settings_.sample_rate) {
      // TODO(melpon): リサンプリングが必要
      PLOG_WARNING << "Needs resampling: " << frame.sample_rate << " Hz to "
                   << settings_.sample_rate << " Hz";
      return;
    }
    if (frame.channels() != 1 && frame.channels() != 2) {
      PLOG_ERROR << "Invalid frame channels: " << frame.channels();
      return;
    }

    // 前フレームからの残りのサンプル数
    auto remain_samples = pcm_buf_.size() / settings_.channels;
    // frame.timestamp から次にエンコードするデータの開始時間を計算する
    auto timestamp =
        frame.timestamp -
        (remain_samples * std::chrono::microseconds(std::chrono::seconds(1)) /
         settings_.sample_rate);

    for (int i = 0; i < frame.samples(); i++) {
      if (frame.channels() == settings_.channels) {
        for (int j = 0; j < frame.channels(); j++) {
          pcm_buf_.push_back(frame.pcm(i, j));
        }
      } else if (frame.channels() == 1 && settings_.channels == 2) {
        // 入力フレームはモノラルだけどエンコーダに渡すのはステレオの場合、同じデータを詰めておく
        pcm_buf_.push_back(frame.pcm(i, 0));
        pcm_buf_.push_back(frame.pcm(i, 0));
      } else /*if (frame.channels() == 2 && settings_.channels == 1)*/ {
        // 入力フレームはステレオだけどエンコーダに渡すのはモノラルの場合、平均の値を詰めておく
        pcm_buf_.push_back((frame.pcm(i, 0) + frame.pcm(i, 1)) / 2);
      }

      // frame_duration_ms 時間分のデータごとにエンコードする
      // 16000 [Hz] * 2 [channels] * (20 [ms] / 1000 [ms])
      if (pcm_buf_.size() < settings_.sample_rate * settings_.channels *
                                settings_.frame_duration_ms / 1000) {
        continue;
      }
      while (true) {
        int n = opus_encode_float(encoder_, pcm_buf_.data(),
                                  pcm_buf_.size() / settings_.channels,
                                  encoded_buf_.data(), encoded_buf_.size());
        // バッファが足りないので増やす
        if (n == OPUS_BUFFER_TOO_SMALL) {
          encoded_buf_.resize(encoded_buf_.size() * 2);
          continue;
        }
        if (n < 0) {
          // 何かエラーが起きた
          PLOG_ERROR << "Failed to opus_encode_float: result=" << n;
          return;
        }
        // 無事エンコードできた
        EncodedAudio encoded;
        encoded.data = CreateAudioBuffer(n);
        encoded.timestamp = timestamp;
        timestamp += std::chrono::milliseconds(settings_.frame_duration_ms);
        memcpy(encoded.data.data(), encoded_buf_.data(), n);
        encoded_bufs_.push_back(std::move(encoded));
        break;
      }
      pcm_buf_.clear();
    }

    for (auto it = encoded_bufs_.begin(); it != encoded_bufs_.end(); ++it) {
      on_encode_(*it);
    }
    encoded_bufs_.clear();
  }

  void SetOnEncode(
      std::function<void(const EncodedAudio&)> on_encode) override {
    on_encode_ = on_encode;
  }

 private:
  Settings settings_;

  OpusEncoder* encoder_ = nullptr;
  std::function<void(const EncodedAudio&)> on_encode_;
  std::vector<float> pcm_buf_;
  std::vector<uint8_t> encoded_buf_;
  std::vector<EncodedAudio> encoded_bufs_;
};

std::shared_ptr<AudioEncoder> CreateOpusAudioEncoder() {
  return std::make_shared<OpusAudioEncoder>();
}
