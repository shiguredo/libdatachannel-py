from datetime import timedelta

import numpy as np

from libdatachannel.codec import (
    AudioCodecType,
    AudioDecoder,
    AudioEncoder,
    AudioFrame,
    EncodedAudio,
    create_opus_audio_decoder,
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


def test_audio_encoder_decoder_simple():
    # まずエンコーダーのみのテスト
    encoder = create_opus_audio_encoder()
    encoder_settings = AudioEncoder.Settings()
    encoder_settings.codec_type = AudioCodecType.OPUS
    encoder_settings.sample_rate = 48000
    encoder_settings.channels = 2
    encoder_settings.bitrate = 64000
    encoder_settings.frame_duration_ms = 20
    assert encoder.init(encoder_settings) is True

    encoded_data = []

    def on_encoded(encoded: EncodedAudio):
        print(f"Encoded data size: {len(encoded.data)}")
        encoded_data.append(encoded)

    encoder.set_on_encode(on_encoded)

    # 簡単なテストデータ
    frame = AudioFrame()
    frame.sample_rate = 48000
    frame.timestamp = timedelta(milliseconds=0)
    frame.pcm = np.zeros((960, 2), dtype=np.float32)

    encoder.encode(frame)
    assert len(encoded_data) == 1

    encoder.release()


def test_audio_decoder_only():
    # デコーダーのみのテスト
    decoder = create_opus_audio_decoder()
    decoder_settings = AudioDecoder.Settings()
    decoder_settings.codec_type = AudioCodecType.OPUS
    decoder_settings.sample_rate = 48000
    decoder_settings.channels = 2
    assert decoder.init(decoder_settings) is True

    # まずエンコーダーでデータを作成
    encoder = create_opus_audio_encoder()
    encoder_settings = AudioEncoder.Settings()
    encoder_settings.codec_type = AudioCodecType.OPUS
    encoder_settings.sample_rate = 48000
    encoder_settings.channels = 2
    encoder_settings.bitrate = 64000
    encoder_settings.frame_duration_ms = 20
    assert encoder.init(encoder_settings) is True

    encoded_data = []

    def on_encoded(encoded: EncodedAudio):
        encoded_data.append(encoded)

    encoder.set_on_encode(on_encoded)

    # シンプルなデータでエンコード
    frame = AudioFrame()
    frame.sample_rate = 48000
    frame.timestamp = timedelta(milliseconds=0)
    frame.pcm = np.zeros((960, 2), dtype=np.float32)

    encoder.encode(frame)
    encoder.release()

    print(f"Encoded data count: {len(encoded_data)}")
    assert len(encoded_data) == 1

    # デコード結果を格納
    decoded_frames = []

    def on_decoded(frame: AudioFrame):
        print(f"Decoded frame: samples={frame.samples()}, channels={frame.channels()}")
        decoded_frames.append(frame)

    decoder.set_on_decode(on_decoded)

    # デコード実行
    print(f"Decoding data with size: {len(encoded_data[0].data)}")
    decoder.decode(encoded_data[0])

    assert len(decoded_frames) == 1
    decoder.release()


def test_audio_encoder_decoder():
    # エンコード→デコードのテスト（ゼロデータで確実に動作することを確認）
    encoder = create_opus_audio_encoder()
    encoder_settings = AudioEncoder.Settings()
    encoder_settings.codec_type = AudioCodecType.OPUS
    encoder_settings.sample_rate = 48000
    encoder_settings.channels = 2
    encoder_settings.bitrate = 64000
    encoder_settings.frame_duration_ms = 20
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
        encoded_data.append(encoded)

    def on_decoded(frame: AudioFrame):
        decoded_frames.append(frame)

    encoder.set_on_encode(on_encoded)
    decoder.set_on_decode(on_decoded)

    # 無音データでテスト
    frame = AudioFrame()
    frame.sample_rate = 48000
    frame.timestamp = timedelta(milliseconds=0)
    frame.pcm = np.zeros((960, 2), dtype=np.float32)

    encoder.encode(frame)
    assert len(encoded_data) == 1

    decoder.decode(encoded_data[0])
    assert len(decoded_frames) == 1

    decoded_frame = decoded_frames[0]
    assert decoded_frame.sample_rate == 48000
    assert decoded_frame.channels() == 2
    assert decoded_frame.samples() == 960
    assert decoded_frame.timestamp == timedelta(milliseconds=0)

    encoder.release()
    decoder.release()
