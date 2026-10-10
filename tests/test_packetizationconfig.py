import pytest

from libdatachannel import RtpPacketizationConfig


def test_basic_initialization():
    cfg = RtpPacketizationConfig(
        ssrc=1234,
        cname="stream1",
        payload_type=111,
        clock_rate=48000,
        video_orientation_id=1,
    )

    assert cfg.ssrc == 1234
    assert cfg.cname == "stream1"
    assert cfg.payload_type == 111
    assert cfg.clock_rate == 48000
    assert cfg.video_orientation_id == 1


def test_default_fields():
    cfg = RtpPacketizationConfig(1, "abc", 96, 90000)

    assert cfg.video_orientation == 0
    assert cfg.mid_id == 0
    assert cfg.mid is None
    assert cfg.rid_id == 0
    assert cfg.rid is None
    assert cfg.playout_delay_id == 0
    assert cfg.playout_delay_min == 0
    assert cfg.playout_delay_max == 0


def test_set_optional_fields():
    cfg = RtpPacketizationConfig(2, "xyz", 97, 8000)
    cfg.mid = "audio"
    cfg.rid = "a1"
    cfg.playout_delay_min = 5
    cfg.playout_delay_max = 20

    assert cfg.mid == "audio"
    assert cfg.rid == "a1"
    assert cfg.playout_delay_min == 5
    assert cfg.playout_delay_max == 20


def test_timestamp_conversion():
    cfg = RtpPacketizationConfig(3, "video", 98, 90000)

    ts = 180000
    sec = cfg.timestamp_to_seconds(ts)
    assert sec == pytest.approx(2.0)

    ts2 = cfg.seconds_to_timestamp(2.0)
    assert ts2 == 180000  # 2s * 90000 = 180000

    # static versions
    assert RtpPacketizationConfig.get_seconds_from_timestamp(90000, 90000) == pytest.approx(1.0)
    assert RtpPacketizationConfig.get_timestamp_from_seconds(1.0, 90000) == 90000


@pytest.mark.parametrize(
    ("seconds", "clock_rate", "expected"),
    [
        # 経過時間 0 は 0 になる
        (0.0, 90000, 0),
        # 1 秒で 1 クロックレート分だけ進む
        (1.0, 90000, 90000),
        # 音声 (48 kHz) も同じ計算になる
        (1.0, 48000, 48000),
        # 端数は四捨五入する (0.9 クロックは 1、 0.45 クロックは 0)
        (0.00001, 90000, 1),
        (0.000005, 90000, 0),
        # 32 bit の範囲を超えると wrap する (9e9 は 2^32 で 2 周する)
        (100000.0, 90000, 410065408),
        # wrap した値も 32 bit に収まる
        (47722.0, 90000, 12704),
        # 上限付近でも 32 bit に収まる
        (89478.0, 48000, 4294944000),
    ],
    ids=[
        "zero_seconds",
        "one_second_video",
        "one_second_audio",
        "round_up",
        "round_down",
        "wrap_after_two_turns",
        "wrap_value",
        "near_upper_bound",
    ],
)
def test_get_timestamp_from_seconds(seconds: float, clock_rate: int, expected: int) -> None:
    """経過秒から RTP timestamp を求めるとき 32 bit で wrap すること

    RTP の timestamp は 32 bit のため、 長時間の配信で範囲を超えても例外に
    ならず wrap した値を返す。 examples/whip.py はこの API で timestamp を
    更新する (毎フレームの差分を足し込むと丸め誤差が累積するため、 最初の
    dts からの経過時間を渡す)。
    """
    assert RtpPacketizationConfig.get_timestamp_from_seconds(seconds, clock_rate) == expected
