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
