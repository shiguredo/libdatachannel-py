import numpy as np

from libdatachannel.codec import VideoFrameBufferBGR888, VideoFrameBufferI420, VideoFrameBufferNV12
from libdatachannel.libyuv import (
    FilterMode,
    FourCC,
    RotationMode,
    convert_to_i420,
    i420_scale,
    i420_to_nv12,
    i420_to_rgb24,
    nv12_scale,
    nv12_to_i420,
    rgb24_to_i420,
)


def test_libyuv_nv12_scale():
    src_width, src_height = 640, 480
    src = VideoFrameBufferNV12.create(src_width, src_height)
    # ignore しないと型エラーになる
    # 参考: https://github.com/wjakob/nanobind/pull/442
    src.y[:, :] = 42  # type: ignore

    dst_width, dst_height = 320, 240
    dst = VideoFrameBufferNV12.create(dst_width, dst_height)
    r = nv12_scale(
        src.y,
        src.uv,
        src.stride_y(),
        src.stride_uv(),
        src.width(),
        src.height(),
        dst.y,
        dst.uv,
        dst.stride_y(),
        dst.stride_uv(),
        dst.width(),
        dst.height(),
        FilterMode.kNone,
    )

    assert r == 0
    assert np.all(dst.y == 42)


def test_libyuv_i420_scale():
    src_width, src_height = 640, 480
    src = VideoFrameBufferI420.create(src_width, src_height)
    src.y[:, :] = 42  # type: ignore

    dst_width, dst_height = 320, 240
    dst = VideoFrameBufferI420.create(dst_width, dst_height)
    r = i420_scale(
        src.y,
        src.u,
        src.v,
        src.stride_y(),
        src.stride_u(),
        src.stride_v(),
        src.width(),
        src.height(),
        dst.y,
        dst.u,
        dst.v,
        dst.stride_y(),
        dst.stride_u(),
        dst.stride_v(),
        dst.width(),
        dst.height(),
        FilterMode.kNone,
    )
    assert r == 0
    assert np.all(dst.y == 42)


def test_libyuv_convert_to_i420():
    nv12 = np.full((640, 480 + 480 // 2), 42, dtype=np.uint8)

    dst = VideoFrameBufferI420.create(640, 480)
    r = convert_to_i420(
        nv12,
        nv12.size,
        dst.y,
        dst.stride_y(),
        dst.u,
        dst.stride_u(),
        dst.v,
        dst.stride_v(),
        0,
        0,
        640,
        480,
        640,
        480,
        RotationMode.kRotate0,
        FourCC.kNV12,
    )
    assert r == 0
    assert np.all(dst.y == 42)


def test_libyuv_nv12_to_i420():
    src = VideoFrameBufferNV12.create(640, 480)
    src.y[:, :] = 42  # type: ignore
    src.uv[:, :] = 128  # type: ignore

    dst = VideoFrameBufferI420.create(640, 480)
    r = nv12_to_i420(
        src.y,
        src.uv,
        src.stride_y(),
        src.stride_uv(),
        dst.y,
        dst.u,
        dst.v,
        dst.stride_y(),
        dst.stride_u(),
        dst.stride_v(),
        src.width(),
        src.height(),
    )

    assert r == 0
    assert np.all(dst.y == 42)
    assert np.all(dst.u == 128)
    assert np.all(dst.v == 128)


def test_libyuv_i420_to_nv12():
    src = VideoFrameBufferI420.create(640, 480)
    src.y[:, :] = 42  # type: ignore
    src.u[:, :] = 128  # type: ignore
    src.v[:, :] = 128  # type: ignore

    dst = VideoFrameBufferNV12.create(640, 480)
    r = i420_to_nv12(
        src.y,
        src.u,
        src.v,
        src.stride_y(),
        src.stride_u(),
        src.stride_v(),
        dst.y,
        dst.uv,
        dst.stride_y(),
        dst.stride_uv(),
        src.width(),
        src.height(),
    )

    assert r == 0
    assert np.all(dst.y == 42)
    assert np.all(dst.uv == 128)


def test_libyuv_rgb24_to_i420():
    src = VideoFrameBufferBGR888.create(640, 480)
    src.bgr[:, :, :] = 42  # type: ignore

    dst = VideoFrameBufferI420.create(640, 480)
    r = rgb24_to_i420(
        src.bgr,
        src.stride(),
        dst.y,
        dst.u,
        dst.v,
        dst.stride_y(),
        dst.stride_u(),
        dst.stride_v(),
        src.width(),
        src.height(),
    )

    assert r == 0
    assert np.all(dst.y == 52)


def test_libyuv_i420_to_rgb24():
    src = VideoFrameBufferI420.create(640, 480)
    src.y[:, :] = 42  # type: ignore
    src.u[:, :] = 128  # type: ignore
    src.v[:, :] = 128  # type: ignore

    dst = VideoFrameBufferBGR888.create(640, 480)
    r = i420_to_rgb24(
        src.y,
        src.u,
        src.v,
        src.stride_y(),
        src.stride_u(),
        src.stride_v(),
        dst.bgr,
        dst.stride(),
        src.width(),
        src.height(),
    )

    assert r == 0
    assert np.all(dst.bgr == np.array([30, 30, 30], dtype=np.uint8))
