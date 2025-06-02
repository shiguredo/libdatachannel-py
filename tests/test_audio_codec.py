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
    # 48000 Hz * 20 ms = 960 サンプル
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


def test_audio_encoder_with_non_zero_data():
    encoder = create_opus_audio_encoder()
    settings = AudioEncoder.Settings()
    settings.codec_type = AudioCodecType.OPUS
    settings.sample_rate = 48000
    settings.channels = 2
    settings.bitrate = 64000
    settings.frame_duration_ms = 20
    assert encoder.init(settings) is True

    def on_encoded(encoded: EncodedAudio):
        assert isinstance(encoded.data, np.ndarray)

    encoder.set_on_encode(on_encoded)

    # 小さなサイン波でテスト
    frame = AudioFrame()
    frame.sample_rate = 48000
    frame.timestamp = timedelta(milliseconds=1000)

    # 440Hz のサイン波を生成
    samples = np.random.randint(48000, 96000)  # ランダムな長さ (1秒～2秒相当)
    # 時間軸を生成 (サンプル数 / サンプリングレート = 秒)
    t = np.arange(samples) / 48000
    # 440Hz のサイン波を生成 (振幅 0.1)
    audio_signal = np.sin(2 * np.pi * 440 * t) * 0.1
    frame.pcm = np.column_stack((audio_signal, audio_signal)).astype(np.float32)

    # ここでセグフォしない
    encoder.encode(frame)

    encoder.release()


@settings(max_examples=1000)
@given(
    samples=st.integers(min_value=48000, max_value=96000),
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


def test_audio_decoder_decode():
    # デコーダーの作成
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

    # ゼロデータでエンコード
    frame = AudioFrame()
    frame.sample_rate = 48000
    frame.timestamp = timedelta(milliseconds=1000)
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
    decoded_frame = decoded_frames[0]
    assert decoded_frame.sample_rate == 48000
    assert decoded_frame.channels() == 2
    assert decoded_frame.samples() == 960
    assert decoded_frame.timestamp == timedelta(milliseconds=1000)
    decoder.release()


def test_audio_encoder_decoder():
    # エンコード→デコードのテスト
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
    frame.timestamp = timedelta(milliseconds=1000)
    frame.pcm = np.zeros((960, 2), dtype=np.float32)

    encoder.encode(frame)
    assert len(encoded_data) == 1

    decoder.decode(encoded_data[0])
    assert len(decoded_frames) == 1

    decoded_frame = decoded_frames[0]
    assert decoded_frame.sample_rate == 48000
    assert decoded_frame.channels() == 2
    assert decoded_frame.samples() == 960
    assert decoded_frame.timestamp == timedelta(milliseconds=1000)

    encoder.release()
    decoder.release()
