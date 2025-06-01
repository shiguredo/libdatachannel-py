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

    PLOG_DEBUG << "OpusAudioEncoder::Init() - sample_rate: " << settings_.sample_rate
               << ", channels: " << settings_.channels
               << ", bitrate: " << settings_.bitrate
               << ", frame_duration_ms: " << settings_.frame_duration_ms;

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
    PLOG_DEBUG << "OpusAudioEncoder initialized successfully";
    return true;
  }
  void Release() override {
    if (encoder_ != nullptr) {
      opus_encoder_destroy(encoder_);
      encoder_ = nullptr;
    }
  }

  void Encode(const AudioFrame& frame) override {
    PLOG_INFO << "OpusAudioEncoder::Encode() called - samples: " << frame.samples() 
              << ", channels: " << frame.channels() 
              << ", sample_rate: " << frame.sample_rate
              << ", settings channels: " << settings_.channels
              << ", on_encode set: " << (on_encode_ ? "true" : "false");
    
    // Debug: Check if we have valid samples
    if (frame.samples() <= 0) {
      PLOG_ERROR << "Invalid number of samples: " << frame.samples();
      return;
    }
    
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

    // Get raw pointer to PCM data
    const float* pcm_data = static_cast<const float*>(frame.pcm.data());
    int channels = frame.channels();
    
    // Debug: Check PCM data pointer
    PLOG_INFO << "PCM data pointer: " << (void*)pcm_data << ", channels: " << channels;
    if (pcm_data == nullptr) {
      PLOG_ERROR << "PCM data pointer is null!";
      return;
    }
    
    // Debug: Check first few samples
    if (frame.samples() > 0) {
      PLOG_INFO << "First PCM sample: " << pcm_data[0];
      if (frame.samples() > 1) {
        PLOG_INFO << "Second PCM sample: " << pcm_data[1];
      }
    }
    
    // Add all samples to buffer
    size_t pcm_buf_size_before = pcm_buf_.size();
    PLOG_INFO << "Starting to add samples to buffer. Current buffer size: " << pcm_buf_size_before;
    
    for (int i = 0; i < frame.samples(); i++) {
      if (frame.channels() == settings_.channels) {
        for (int j = 0; j < frame.channels(); j++) {
          // Access data correctly: row-major order (samples x channels)
          float sample = pcm_data[i * channels + j];
          pcm_buf_.push_back(sample);
          
          // Debug first few samples
          if (i < 2 && j == 0) {
            PLOG_DEBUG << "Sample[" << i << "][" << j << "] = " << sample;
          }
        }
      } else if (frame.channels() == 1 && settings_.channels == 2) {
        // 入力フレームはモノラルだけどエンコーダに渡すのはステレオの場合、同じデータを詰めておく
        float sample = pcm_data[i];
        pcm_buf_.push_back(sample);
        pcm_buf_.push_back(sample);
        
        if (i < 2) {
          PLOG_DEBUG << "Mono to Stereo - Sample[" << i << "] = " << sample;
        }
      } else /*if (frame.channels() == 2 && settings_.channels == 1)*/ {
        // 入力フレームはステレオだけどエンコーダに渡すのはモノラルの場合、平均の値を詰めておく
        float avg = (pcm_data[i * 2 + 0] + pcm_data[i * 2 + 1]) / 2;
        pcm_buf_.push_back(avg);
        
        if (i < 2) {
          PLOG_DEBUG << "Stereo to Mono - Sample[" << i << "] = " << avg;
        }
      }
    }
    
    PLOG_INFO << "Added " << (pcm_buf_.size() - pcm_buf_size_before) 
              << " samples to buffer. Total buffer size: " << pcm_buf_.size();

    // Encode when we have enough samples
    // frame_duration_ms 時間分のデータごとにエンコードする
    // Calculate required frame count (not total float count)
    size_t required_frames = settings_.sample_rate * settings_.frame_duration_ms / 1000;
    // Total floats needed in buffer = frames * channels
    size_t required_floats = required_frames * settings_.channels;
    
    PLOG_INFO << "Required frames for encoding: " << required_frames
              << " (frame_duration_ms: " << settings_.frame_duration_ms << ")"
              << ", required floats in buffer: " << required_floats
              << ", pcm_buf size: " << pcm_buf_.size();
    
    int encode_count = 0;
    while (pcm_buf_.size() >= required_floats) {
      PLOG_INFO << "Calling opus_encode_float with " 
                << required_frames
                << " frames, pcm_buf.data() size: " << pcm_buf_.size();
      
      // opus_encode_float expects frame count, not total sample count
      int n = opus_encode_float(encoder_, pcm_buf_.data(),
                                required_frames,
                                encoded_buf_.data(), encoded_buf_.size());
      // バッファが足りないので増やす
      if (n == OPUS_BUFFER_TOO_SMALL) {
        PLOG_DEBUG << "OPUS_BUFFER_TOO_SMALL - resizing buffer from " 
                   << encoded_buf_.size() << " to " << encoded_buf_.size() * 2;
        encoded_buf_.resize(encoded_buf_.size() * 2);
        continue;
      }
      if (n < 0) {
        // 何かエラーが起きた
        PLOG_ERROR << "Failed to opus_encode_float: result=" << n;
        return;
      }
      // 無事エンコードできた
      PLOG_INFO << "opus_encode_float succeeded - encoded " << n << " bytes";
      
      EncodedAudio encoded;
      encoded.data = CreateAudioBuffer(n);
      encoded.timestamp = timestamp;
      timestamp += std::chrono::milliseconds(settings_.frame_duration_ms);
      memcpy(encoded.data.data(), encoded_buf_.data(), n);
      encoded_bufs_.push_back(std::move(encoded));
      encode_count++;
      
      // Remove encoded samples from buffer
      pcm_buf_.erase(pcm_buf_.begin(), 
                     pcm_buf_.begin() + required_floats);
    }
    
    if (encode_count > 0) {
      PLOG_DEBUG << "Encoded " << encode_count << " frames in this call";
    }

    for (auto it = encoded_bufs_.begin(); it != encoded_bufs_.end(); ++it) {
      PLOG_INFO << "Calling on_encode callback with " << it->data.size() << " bytes";
      if (on_encode_) {
        on_encode_(*it);
      } else {
        PLOG_ERROR << "on_encode callback is not set!";
      }
    }
    if (!encoded_bufs_.empty()) {
      PLOG_INFO << "Sent " << encoded_bufs_.size() << " encoded frames to callback";
    } else {
      PLOG_WARNING << "No encoded frames to send after Encode() call";
    }
    encoded_bufs_.clear();
  }

  void SetOnEncode(
      std::function<void(const EncodedAudio&)> on_encode) override {
    on_encode_ = on_encode;
    PLOG_INFO << "OpusAudioEncoder::SetOnEncode() - callback " 
              << (on_encode ? "set" : "cleared");
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
