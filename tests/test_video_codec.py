import os
import time
from datetime import timedelta

import numpy as np
import pytest

from libdatachannel import (
    NalUnit,
)
from libdatachannel.codec import (
    EncodedImage,
    ImageFormat,
    VideoCodecType,
    VideoDecoder,
    VideoEncoder,
    VideoFrame,
    VideoFrameBufferBGR888,
    VideoFrameBufferI420,
    VideoFrameBufferNV12,
    create_aom_video_decoder,
    create_aom_video_encoder,
    create_openh264_video_decoder,
    create_openh264_video_encoder,
    create_videotoolbox_video_encoder,
)


def test_video_frame_buffer():
    # I420
    i420_buffer = VideoFrameBufferI420.create(640, 480)
    assert i420_buffer.width() == 640
    assert i420_buffer.height() == 480
    assert i420_buffer.stride_y() == 640
    assert i420_buffer.stride_u() == 320
    assert i420_buffer.stride_v() == 320
    assert isinstance(i420_buffer.y, np.ndarray)
    assert isinstance(i420_buffer.u, np.ndarray)
    assert isinstance(i420_buffer.v, np.ndarray)
    i420_buffer.y[0] = 255
    i420_buffer.y = np.zeros((100, 100), dtype=np.uint8)
    assert i420_buffer.width() == 100

    # NV12
    nv12_buffer = VideoFrameBufferNV12.create(640, 480)
    assert nv12_buffer.width() == 640
    assert nv12_buffer.height() == 480
    assert nv12_buffer.stride_y() == 640
    assert nv12_buffer.stride_uv() == 640
    assert isinstance(nv12_buffer.y, np.ndarray)
    assert isinstance(nv12_buffer.uv, np.ndarray)

    # BGR888
    bgr888_buffer = VideoFrameBufferBGR888.create(640, 480)
    assert bgr888_buffer.width() == 640
    assert bgr888_buffer.height() == 480
    assert bgr888_buffer.stride() == 640 * 3
    assert isinstance(bgr888_buffer.bgr, np.ndarray)


def test_video_frame_properties():
    frame = VideoFrame()

    # format による width()/height() の分岐
    frame.i420_buffer = VideoFrameBufferI420.create(640, 360)
    frame.nv12_buffer = VideoFrameBufferNV12.create(1280, 720)
    frame.bgr888_buffer = VideoFrameBufferBGR888.create(1920, 1080)

    frame.format = ImageFormat.I420
    assert frame.width() == 640
    assert frame.height() == 360

    frame.format = ImageFormat.NV12
    assert frame.width() == 1280
    assert frame.height() == 720

    frame.format = ImageFormat.BGR888
    assert frame.width() == 1920
    assert frame.height() == 1080

    # timestamp, rid
    ts = timedelta(microseconds=1234567)
    frame.timestamp = ts
    assert frame.timestamp == ts

    frame.rid = "abc"
    assert frame.rid == "abc"

    # base_width, base_height
    frame.base_width = 999
    frame.base_height = 888
    assert frame.base_width == 999
    assert frame.base_height == 888


def test_encoded_image_properties():
    img = EncodedImage()

    img.data = np.array([], dtype=np.uint8)
    assert img.data.size == 0
    img.data = np.array([1, 2, 3], dtype=np.uint8)
    assert img.data.size == 3

    ts = timedelta(microseconds=555000)
    img.timestamp = ts
    assert img.timestamp == ts

    img.rid = "r0"
    assert img.rid == "r0"


def test_openh264():
    openh264 = os.environ.get("OPENH264_PATH")
    assert openh264 is not None
    encoder = create_openh264_video_encoder(openh264)
    settings = VideoEncoder.Settings()
    settings.codec_type = VideoCodecType.H264
    settings.width = 640
    settings.height = 480
    settings.bitrate = 1000000
    success = encoder.init(settings)
    assert success
    frame = VideoFrame()
    frame.i420_buffer = VideoFrameBufferI420.create(640, 360)
    frame.format = ImageFormat.I420
    frame.base_width = 640
    frame.base_height = 480
    frame.timestamp = timedelta(microseconds=1234567)

    on_encode_called = False

    def on_encode(encoded_image):
        nonlocal on_encode_called
        assert encoded_image.data.size > 0
        assert encoded_image.timestamp == frame.timestamp
        assert encoded_image.rid == frame.rid
        on_encode_called = True

    encoder.set_on_encode(on_encode)
    encoder.encode(frame)
    encoder.release()
    assert on_encode_called


@pytest.mark.skipif(
    os.environ.get("ENABLE_VIDEOTOOLBOX") is None, reason="macOS の場合だけ実行する"
)
def test_videotoolbox():
    encoder = create_videotoolbox_video_encoder()
    settings = VideoEncoder.Settings()
    settings.codec_type = VideoCodecType.H264
    settings.width = 640
    settings.height = 480
    settings.bitrate = 1000000
    success = encoder.init(settings)
    assert success
    frame = VideoFrame()
    frame.nv12_buffer = VideoFrameBufferNV12.create(640, 360)
    frame.format = ImageFormat.NV12
    frame.base_width = 640
    frame.base_height = 480
    frame.timestamp = timedelta(microseconds=1234567)

    on_encode_called = False

    def on_encode(encoded_image):
        nonlocal on_encode_called
        assert encoded_image.data.size > 0
        assert encoded_image.timestamp == frame.timestamp
        assert encoded_image.rid == frame.rid
        on_encode_called = True

    encoder.set_on_encode(on_encode)
    encoder.encode(frame)
    time.sleep(1)
    encoder.release()
    assert on_encode_called


def test_aom():
    encoder = create_aom_video_encoder()
    settings = VideoEncoder.Settings()
    settings.codec_type = VideoCodecType.H264
    settings.width = 640
    settings.height = 480
    settings.bitrate = 1000000
    success = encoder.init(settings)
    assert success
    frame = VideoFrame()
    frame.i420_buffer = VideoFrameBufferI420.create(640, 360)
    frame.format = ImageFormat.I420
    frame.base_width = 640
    frame.base_height = 480
    frame.timestamp = timedelta(microseconds=1234567)

    on_encode_called = False

    def on_encode(encoded_image):
        nonlocal on_encode_called
        assert encoded_image.data.size > 0
        assert encoded_image.timestamp == frame.timestamp
        assert encoded_image.rid == frame.rid
        on_encode_called = True

    encoder.set_on_encode(on_encode)
    encoder.encode(frame)
    encoder.release()
    assert on_encode_called


def test_openh264_encode_decode():
    """OpenH264を使用してnumpyで生成した映像をエンコード・デコードするテスト"""
    openh264 = os.environ.get("OPENH264_PATH")
    assert openh264 is not None

    # エンコーダーの初期化
    encoder = create_openh264_video_encoder(openh264)
    settings = VideoEncoder.Settings()
    settings.codec_type = VideoCodecType.H264
    settings.width = 320
    settings.height = 240
    settings.bitrate = 500000
    settings.fps = 30
    success = encoder.init(settings)
    assert success

    # テスト用の映像フレームを生成（グラデーションパターン）
    width, height = 320, 240
    i420_buffer = VideoFrameBufferI420.create(width, height)

    # Y平面：グラデーション（左から右へ明るくなる）
    y_gradient = np.linspace(16, 235, width, dtype=np.uint8)
    y_plane = y_gradient[np.newaxis, :].repeat(height, axis=0)
    i420_buffer.y[:, :] = y_plane

    # U,V平面：単色（グレー）
    i420_buffer.u[:, :] = 128
    i420_buffer.v[:, :] = 128

    frame = VideoFrame()
    frame.i420_buffer = i420_buffer
    frame.format = ImageFormat.I420
    frame.base_width = width
    frame.base_height = height
    frame.timestamp = timedelta(microseconds=1000)

    encoded_frames = []

    def on_encode(encoded_image):
        encoded_frames.append(encoded_image)
        assert encoded_image.data.size > 0

    encoder.set_on_encode(on_encode)
    encoder.force_intra_next_frame()  # キーフレームを生成
    encoder.encode(frame)

    # 2フレーム目も送信
    frame.timestamp = timedelta(microseconds=1000 + 33333)  # 30fps
    encoder.encode(frame)
    encoder.release()

    assert len(encoded_frames) == 2
    print(f"Encoded {len(encoded_frames)} frames")

    # デコーダーでデコード
    decoder = create_openh264_video_decoder(openh264)
    decoder_settings = VideoDecoder.Settings()
    decoder_settings.codec_type = VideoCodecType.H264
    success = decoder.init(decoder_settings)
    assert success

    decoded_frames = []

    def on_decode(video_frame):
        nonlocal decoded_frames
        decoded_frames.append(video_frame)
        print(f"Decoded frame: {video_frame.width()}x{video_frame.height()}")

    decoder.set_on_decode(on_decode)

    # すべてのエンコード済みフレームをデコード
    for i, encoded in enumerate(encoded_frames):
        encoded_image = EncodedImage()
        encoded_image.data = encoded.data
        encoded_image.timestamp = encoded.timestamp
        print(f"Decoding frame {i + 1}: {encoded_image.data.size} bytes...")
        decoder.decode(encoded_image)

    decoder.release()

    assert len(decoded_frames) == 2
    assert decoded_frames[0].width() == width
    assert decoded_frames[0].height() == height
    assert decoded_frames[0].format == ImageFormat.I420
    assert decoded_frames[0].timestamp == timedelta(microseconds=1000)
    assert decoded_frames[1].width() == width
    assert decoded_frames[1].height() == height
    assert decoded_frames[1].format == ImageFormat.I420
    assert decoded_frames[1].timestamp == timedelta(microseconds=1000 + 33333)

    # デコードされたフレームの内容を簡単に検証
    # Y平面の最初と最後の値がグラデーションになっているか確認
    decoded_y = decoded_frames[0].i420_buffer.y.reshape(height, width)
    assert decoded_y[0, 0] < decoded_y[0, -1]  # 左端より右端の方が明るい


def test_openh264_with_nal_units():
    """OpenH264とNALユニット解析を使用したテスト"""
    openh264 = os.environ.get("OPENH264_PATH")
    assert openh264 is not None

    # エンコーダーの初期化
    encoder = create_openh264_video_encoder(openh264)
    settings = VideoEncoder.Settings()
    settings.codec_type = VideoCodecType.H264
    settings.width = 320
    settings.height = 240
    settings.bitrate = 500000
    settings.fps = 30
    success = encoder.init(settings)
    assert success

    # テスト用の映像フレームを生成（グラデーションパターン）
    width, height = 320, 240
    i420_buffer = VideoFrameBufferI420.create(width, height)

    # Y平面：グラデーション（左から右へ明るくなる）
    y_gradient = np.linspace(16, 235, width, dtype=np.uint8)
    y_plane = y_gradient[np.newaxis, :].repeat(height, axis=0)
    i420_buffer.y[:, :] = y_plane

    # U,V平面：単色（グレー）
    i420_buffer.u[:, :] = 128
    i420_buffer.v[:, :] = 128

    frame = VideoFrame()
    frame.i420_buffer = i420_buffer
    frame.format = ImageFormat.I420
    frame.base_width = width
    frame.base_height = height
    frame.timestamp = timedelta(microseconds=1000)

    encoded_frames = []

    def on_encode(encoded_image):
        encoded_frames.append(encoded_image)

    encoder.set_on_encode(on_encode)
    encoder.force_intra_next_frame()  # キーフレームを生成
    encoder.encode(frame)

    # 2フレーム目も送信
    frame.timestamp = timedelta(microseconds=1000 + 33333)
    encoder.encode(frame)
    encoder.release()

    assert len(encoded_frames) == 2
    print(f"Encoded {len(encoded_frames)} frames")

    # 最初のフレームのNALユニットを解析
    first_encoded = encoded_frames[0].data
    print(f"First frame size: {first_encoded.size} bytes")

    # NALユニットを分割して作成
    nal_units_byte = bytes(first_encoded).split(b"\x00\x00\x00\x01")[1:]
    nal_units = list(map(lambda b: NalUnit(b), nal_units_byte))

    # NALユニットのヘッダー情報を検証
    assert len(nal_units) == 3
    assert nal_units[0].unit_type() == 7  # SPS
    assert nal_units[1].unit_type() == 8  # PPS
    assert nal_units[2].unit_type() == 5  # IDR

    # デコーダーでデコード
    decoder = create_openh264_video_decoder(openh264)
    decoder_settings = VideoDecoder.Settings()
    decoder_settings.codec_type = VideoCodecType.H264
    success = decoder.init(decoder_settings)
    assert success

    decoded_frames = []

    def on_decode(video_frame):
        nonlocal decoded_frames
        decoded_frames.append(video_frame)
        print(f"Decoded frame: {video_frame.width()}x{video_frame.height()}")

    decoder.set_on_decode(on_decode)

    # すべてのエンコード済みフレームをデコード
    for i, encoded in enumerate(encoded_frames):
        encoded_image = EncodedImage()
        encoded_image.data = encoded.data
        encoded_image.timestamp = encoded.timestamp
        print(f"Decoding frame {i + 1}: {encoded_image.data.size} bytes...")
        decoder.decode(encoded_image)

    decoder.release()

    # デコード結果の検証
    assert len(decoded_frames) == 2
    assert decoded_frames[0].width() == width
    assert decoded_frames[0].height() == height
    assert decoded_frames[0].timestamp == timedelta(microseconds=1000)
    assert decoded_frames[1].width() == width
    assert decoded_frames[1].height() == height
    assert decoded_frames[1].timestamp == timedelta(microseconds=1000 + 33333)

    # デコードされたフレームの内容を簡単に検証
    decoded_y = decoded_frames[0].i420_buffer.y.reshape(height, width)
    assert decoded_y[0, 0] < decoded_y[0, -1]  # 左端より右端の方が明るい


def test_aom_encode_decode():
    """AOMを使用してnumpyで生成した映像をエンコード・デコードするテスト"""
    # エンコーダーの初期化
    encoder = create_aom_video_encoder()
    settings = VideoEncoder.Settings()
    settings.codec_type = VideoCodecType.AV1
    settings.width = 320
    settings.height = 240
    settings.bitrate = 500000
    settings.fps = 30
    success = encoder.init(settings)
    assert success

    # テスト用の映像フレームを生成（グラデーションパターン）
    width, height = 320, 240
    i420_buffer = VideoFrameBufferI420.create(width, height)

    # Y平面：グラデーション（左から右へ明るくなる）
    y_gradient = np.linspace(16, 235, width, dtype=np.uint8)
    y_plane = y_gradient[np.newaxis, :].repeat(height, axis=0)
    i420_buffer.y[:, :] = y_plane

    # U,V平面：単色（グレー）
    i420_buffer.u[:, :] = 128
    i420_buffer.v[:, :] = 128

    frame = VideoFrame()
    frame.i420_buffer = i420_buffer
    frame.format = ImageFormat.I420
    frame.base_width = width
    frame.base_height = height
    frame.timestamp = timedelta(microseconds=1000)

    encoded_frames = []

    def on_encode(encoded_image):
        encoded_frames.append(encoded_image)
        assert encoded_image.data.size > 0

    encoder.set_on_encode(on_encode)
    encoder.force_intra_next_frame()  # キーフレームを生成
    encoder.encode(frame)

    # 2フレーム目も送信
    frame.timestamp = timedelta(microseconds=1000 + 33333)  # 30fps
    encoder.encode(frame)
    encoder.release()

    assert len(encoded_frames) == 2
    print(f"Encoded {len(encoded_frames)} frames")

    assert encoded_frames[0].timestamp == timedelta(microseconds=1000)
    assert encoded_frames[1].timestamp == timedelta(microseconds=1000 + 33333)

    # デコーダーでデコード
    decoder = create_aom_video_decoder()
    decoder_settings = VideoDecoder.Settings()
    decoder_settings.codec_type = VideoCodecType.AV1
    success = decoder.init(decoder_settings)
    assert success

    decoded_frames = []

    def on_decode(video_frame):
        nonlocal decoded_frames
        decoded_frames.append(video_frame)
        print(f"Decoded frame: {video_frame.width()}x{video_frame.height()}")

    decoder.set_on_decode(on_decode)

    # すべてのエンコード済みフレームをデコード
    for i, encoded in enumerate(encoded_frames):
        encoded_image = EncodedImage()
        encoded_image.data = encoded.data
        encoded_image.timestamp = encoded.timestamp
        print(f"Decoding frame {i + 1}: {encoded_image.data.size} bytes...")
        decoder.decode(encoded_image)

    decoder.release()

    assert len(decoded_frames) == 2
    assert decoded_frames[0].width() == width
    assert decoded_frames[0].height() == height
    assert decoded_frames[0].format == ImageFormat.I420
    assert decoded_frames[0].timestamp == timedelta(microseconds=1000)
    assert decoded_frames[1].width() == width
    assert decoded_frames[1].height() == height
    assert decoded_frames[1].format == ImageFormat.I420
    assert decoded_frames[1].timestamp == timedelta(microseconds=1000 + 33333)

    # デコードされたフレームの内容を簡単に検証
    # Y平面の最初と最後の値がグラデーションになっているか確認
    decoded_y = decoded_frames[0].i420_buffer.y.reshape(height, width)
    assert decoded_y[0, 0] < decoded_y[0, -1]  # 左端より右端の方が明るい
