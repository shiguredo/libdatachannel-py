from datetime import timedelta

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from libdatachannel.codec import (
    AudioCodecType,
    AudioEncoder,
    AudioFrame,
    EncodedAudio,
    create_opus_audio_encoder,
)


@settings(max_examples=1000)
@given(
    samples=st.integers(min_value=0, max_value=96000),
    # https://www.rfc-editor.org/rfc/rfc6716#section-2.1.1
    # Opus supports all bitrates from 6 kbit/s to 510 kbit/s.
    bitrate=st.integers(min_value=6000, max_value=510000),
    # https://www.rfc-editor.org/rfc/rfc6716#section-2.1.4
    # Opus can encode frames of 2.5, 5, 10, 20, 40, or 60 ms.  It can also combine multiple frames into packets of up to 120 ms.
    # TODO: 2.5ms は現時点では対応しない
    # frame_duration_ms=st.sampled_from([2.5, 5, 10, 20, 40, 60, 120]),
    frame_duration_ms=st.sampled_from([5, 10, 20, 40, 60, 120]),
)
def test_prop_audio_encoder(samples, bitrate, frame_duration_ms):
    """PBT を利用したエンコーダーのテスト"""
    encoder = create_opus_audio_encoder()
    encoder_settings = AudioEncoder.Settings()
    encoder_settings.codec_type = AudioCodecType.OPUS
    encoder_settings.sample_rate = 48000
    encoder_settings.channels = 2
    encoder_settings.bitrate = bitrate
    encoder_settings.frame_duration_ms = frame_duration_ms
    print(
        f"Testing with samples={samples}, bitrate={bitrate}, frame_duration_ms={frame_duration_ms}"
    )
    assert encoder.init(encoder_settings) is True

    encoded_count = 0

    def on_encoded(encoded: EncodedAudio):
        nonlocal encoded_count
        assert isinstance(encoded.data, np.ndarray)
        assert len(encoded.data) > 0
        encoded_count += 1

    encoder.set_on_encode(on_encoded)

    frame = AudioFrame()
    frame.sample_rate = 48000
    frame.timestamp = timedelta(milliseconds=1000)

    # 時間軸を生成 (サンプル数 / サンプリングレート = 秒)
    t = np.arange(samples) / 48000
    # 440Hz のサイン波を生成 (振幅 0.1)
    audio_signal = np.sin(2 * np.pi * 440 * t) * 0.1
    frame.pcm = np.column_stack((audio_signal, audio_signal)).astype(np.float32)

    # エンコード実行
    encoder.encode(frame)

    # frame_duration_ms に基づいて期待されるフレーム数を計算
    samples_per_frame = int(48000 * frame_duration_ms / 1000)
    expected_frames = samples // samples_per_frame
    assert encoded_count == expected_frames

    encoder.release()
