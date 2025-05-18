import os
import time
from datetime import timedelta

import numpy as np
import pytest

from libdatachannel import (
    EncodedImage,
    ImageFormat,
    VideoCodecType,
    VideoEncoder,
    VideoFrame,
    VideoFrameBufferBGR888,
    VideoFrameBufferI420,
    VideoFrameBufferNV12,
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

    on_encoded_called = False

    def on_encoded(encoded_image):
        nonlocal on_encoded_called
        assert encoded_image.data.size > 0
        assert encoded_image.timestamp == frame.timestamp
        assert encoded_image.rid == frame.rid
        on_encoded_called = True

    encoder.set_on_encoded(on_encoded)
    encoder.encode(frame)
    encoder.release()
    assert on_encoded_called


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

    on_encoded_called = False

    def on_encoded(encoded_image):
        nonlocal on_encoded_called
        assert encoded_image.data.size > 0
        assert encoded_image.timestamp == frame.timestamp
        assert encoded_image.rid == frame.rid
        on_encoded_called = True

    encoder.set_on_encoded(on_encoded)
    encoder.encode(frame)
    time.sleep(1)
    encoder.release()
    assert on_encoded_called
