from datetime import timedelta

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from libdatachannel.codec import (
    AudioCodecType,
    AudioDecoder,
    AudioEncoder,
    AudioFrame,
    EncodedAudio,
    create_opus_audio_decoder,
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
def test_prop_audio_encode_decode(samples, bitrate, frame_duration_ms):
    """PBT を利用したエンコーダー・デコーダーのテスト"""
    encoder = create_opus_audio_encoder()
    encoder_settings = AudioEncoder.Settings()
    encoder_settings.codec_type = AudioCodecType.OPUS
    encoder_settings.sample_rate = 48000
    encoder_settings.channels = 2
    encoder_settings.bitrate = bitrate
    encoder_settings.frame_duration_ms = frame_duration_ms
    assert encoder.init(encoder_settings) is True

    decoder = create_opus_audio_decoder()
    decoder_settings = AudioDecoder.Settings()
    decoder_settings.codec_type = AudioCodecType.OPUS
    decoder_settings.sample_rate = 48000
    decoder_settings.channels = 2
    assert decoder.init(decoder_settings) is True

    encoded_data = []
    decoded_frames = []

    def on_encoded(encoded: EncodedAudio):
        assert isinstance(encoded.data, np.ndarray)
        assert len(encoded.data) > 0
        encoded_data.append(encoded)

    def on_decoded(frame: AudioFrame):
        decoded_frames.append(frame)

    encoder.set_on_encode(on_encoded)
    decoder.set_on_decode(on_decoded)

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
    assert len(encoded_data) == expected_frames

    # デコード実行
    for encoded in encoded_data:
        decoder.decode(encoded)

    # デコードされたフレーム数の確認
    assert len(decoded_frames) == expected_frames

    # デコードされたフレームのプロパティを確認
    for i, decoded_frame in enumerate(decoded_frames):
        assert decoded_frame.sample_rate == 48000
        assert decoded_frame.channels() == 2
        assert decoded_frame.samples() == samples_per_frame
        # タイムスタンプの確認
        expected_timestamp = timedelta(milliseconds=1000 + i * frame_duration_ms)
        assert decoded_frame.timestamp == expected_timestamp

    encoder.release()
    decoder.release()
