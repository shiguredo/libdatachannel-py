"""examples/rtp_timestamp.py のテスト

RTP timestamp が 32 bit で wrap すること、 初期値 (乱数) が維持されること、
毎フレームの差分を足し込む方式のように丸め誤差が累積しないことを確認する。
"""

import importlib.util
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parent.parent / "examples" / "rtp_timestamp.py"
_SPEC = importlib.util.spec_from_file_location("rtp_timestamp", _MODULE_PATH)
assert _SPEC is not None
rtp_timestamp = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(rtp_timestamp)

compute_rtp_timestamp = rtp_timestamp.compute_rtp_timestamp
RtpTimestampCalculator = rtp_timestamp.RtpTimestampCalculator

# 映像のクロックレート (90 kHz) と Opus のクロックレート (48 kHz)
VIDEO_CLOCK_RATE = 90000
AUDIO_CLOCK_RATE = 48000


@pytest.mark.parametrize(
    ("start_timestamp", "dts_usec", "clock_rate", "expected"),
    [
        # 最初のフレーム (経過時間 0) は初期値のまま
        (12345, 0, VIDEO_CLOCK_RATE, 12345),
        # 初期値が大きくても 0 にはならない
        (0xFFFFFFFF, 0, VIDEO_CLOCK_RATE, 0xFFFFFFFF),
        # 1 秒で 1 クロックレート分だけ進む
        (1000, 1_000_000, VIDEO_CLOCK_RATE, 91000),
        # 音声 (48 kHz) も同じ計算になる
        (1000, 1_000_000, AUDIO_CLOCK_RATE, 49000),
        # 2 秒で 2 クロックレート分だけ進む
        (0, 2_000_000, VIDEO_CLOCK_RATE, 180000),
        # 端数は四捨五入する (499.95 クロックは 500)
        (0, 5_555, VIDEO_CLOCK_RATE, 500),
        # 初期値から 32 bit を超えると wrap する
        (0xFFFFFFFF, 1_000_000, VIDEO_CLOCK_RATE, 89999),
        # 32 bit の上限を超えても例外にならず 32 bit に収まる
        (0xFFFFFF00, 1_000_000, VIDEO_CLOCK_RATE, 0x15E90),
        # 音声でも wrap する
        (0xFFFFFFFF, 1_000_000, AUDIO_CLOCK_RATE, 47999),
    ],
    ids=[
        "first_frame_keeps_start",
        "first_frame_with_large_start",
        "one_second_video",
        "one_second_audio",
        "two_seconds_elapsed",
        "round_fraction",
        "wrap_video",
        "wrap_with_large_start",
        "wrap_audio",
    ],
)
def test_compute_rtp_timestamp(
    start_timestamp: int,
    dts_usec: int,
    clock_rate: int,
    expected: int,
) -> None:
    """最初の dts からの経過時間から 32 bit に収まる RTP timestamp を返すこと

    初期値を維持したまま wrap させ、 常に 32 bit の範囲に収まる値を返す。
    """
    assert compute_rtp_timestamp(start_timestamp, 0, dts_usec, clock_rate) == expected


def test_compute_rtp_timestamp_with_non_zero_first_dts() -> None:
    """最初の dts が 0 でない場合も、 そこからの経過時間で計算すること

    dts は 0 から始まるとは限らないため、 最初の dts を基準にする。
    """
    first_dts_usec = 5_000_000

    # 1 秒経過で 90000 進む (初期値 1000 を維持する)
    assert compute_rtp_timestamp(1000, first_dts_usec, 6_000_000, VIDEO_CLOCK_RATE) == 91000
    # 最初のフレームでは初期値のまま
    assert compute_rtp_timestamp(1000, first_dts_usec, first_dts_usec, VIDEO_CLOCK_RATE) == 1000


def test_compute_rtp_timestamp_does_not_accumulate_rounding_error() -> None:
    """フレームを連続で渡しても丸め誤差が累積しないこと

    33.333 ms 間隔のフレームを 1000 枚送ると、 1 枚ごとの切り捨てを足し込む方式は
    90 kHz 換算で 1 枚あたり 0.97 クロックを失い、 1000 枚で 970 クロックずれる。
    基準からの経過時間で計算するため、 ずれは 1 クロック未満に収まる
    (ずれはフレーム間隔によって変わり、 長時間の配信では無視できない大きさになる)。
    """
    frame_interval_usec = 33_333
    frames = 1000
    calculator = RtpTimestampCalculator(0, VIDEO_CLOCK_RATE)

    # 1 枚ごとの切り捨てを足し込む方式 (従来の計算)
    accumulated = frame_interval_usec * VIDEO_CLOCK_RATE // 1_000_000 * frames
    # フレーム列を連続で渡す (実装と同じ経路)
    computed = 0
    for index in range(frames + 1):
        computed = calculator.update(frame_interval_usec * index)

    assert accumulated == 2_999_000
    assert computed == 2_999_970
    assert computed - accumulated == 970


def test_rtp_timestamp_calculator_keeps_start_timestamp_on_first_frame() -> None:
    """最初のフレームでは初期値をそのまま返すこと

    RTP timestamp の初期値は乱数であるため、 0 から始めてはならない。
    """
    start_timestamp = 0xC0FFEE12
    calculator = RtpTimestampCalculator(start_timestamp, VIDEO_CLOCK_RATE)

    assert calculator.update(123_456) == start_timestamp
    # 2 枚目は初期値から進む
    assert calculator.update(123_456 + 1_000_000) == start_timestamp + 90_000
