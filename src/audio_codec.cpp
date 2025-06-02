#include "audio_codec.h"

audio_buffer_type CreateAudioBuffer(int size) {
  auto ptr = new uint8_t[size]();
  nanobind::capsule owner(
      ptr, [](void* ptr) noexcept { delete[] static_cast<uint8_t*>(ptr); });
  return audio_buffer_type(ptr, {(size_t)size}, owner, {(int64_t)1});
}

pcm_float_type CreatePCMFloat(int samples, int channels) {
  auto ptr = new float[samples * channels]();
  nanobind::capsule owner(
      ptr, [](void* ptr) noexcept { delete[] static_cast<float*>(ptr); });
  return pcm_float_type(ptr, {(size_t)samples, (size_t)channels}, owner,
                        {(int64_t)channels, 1});
}

int AudioFrame::channels() const {
  return pcm.shape(1);
}
int AudioFrame::samples() const {
  return pcm.shape(0);
}
