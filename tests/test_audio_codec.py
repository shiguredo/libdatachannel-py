from datetime import timedelta

import numpy as np

from libdatachannel import (
    AudioCodecType,
    AudioEncoder,
    AudioFrame,
    EncodedAudio,
    create_opus_audio_encoder,
)


def test_audio_encoder_encode():
    encoder = create_opus_audio_encoder()

    settings = AudioEncoder.Settings()
    settings.codec_type = AudioCodecType.OPUS
    settings.sample_rate = 48000
    settings.channels = 2
    settings.bitrate = 64000
    settings.frame_duration_ms = 20

    assert encoder.init(settings) is True

    callback_timestamps = []

    def on_encoded(encoded: EncodedAudio):
        assert isinstance(encoded.data, np.ndarray)
        callback_timestamps.append(encoded.timestamp)

    encoder.set_on_encode(on_encoded)

    frame = AudioFrame()
    frame.sample_rate = 48000
    frame.timestamp = timedelta(milliseconds=1000)
    # 48000 Hz * 20 ms = 960 samples
    frame.pcm = np.zeros((960, 2), dtype=np.float32)

    encoder.encode(frame)

    assert callback_timestamps == [timedelta(milliseconds=1000)]

    # 10ms → 20ms でエンコード
    frame = AudioFrame()
    frame.sample_rate = 48000
    frame.timestamp = timedelta(milliseconds=1000)
    frame.pcm = np.zeros((480, 2), dtype=np.float32)

    encoder.encode(frame)

    frame.timestamp = timedelta(milliseconds=1100)  # わざと大きくずらす
    frame.pcm = np.zeros((960, 2), dtype=np.float32)

    encoder.encode(frame)

    # 既存のフレームが 10ms あって、1100 だったので 1100 - 10 の 1090 になるはず
    assert callback_timestamps == [timedelta(milliseconds=1000), timedelta(milliseconds=1090)]

    # 10ms のデータを渡すと追加でエンコードされるはず
    frame.timestamp = timedelta(milliseconds=1120)
    frame.pcm = np.zeros((480, 2), dtype=np.float32)

    encoder.encode(frame)

    assert callback_timestamps == [
        timedelta(milliseconds=1000),
        timedelta(milliseconds=1090),
        timedelta(milliseconds=1110),
    ]

    encoder.release()
