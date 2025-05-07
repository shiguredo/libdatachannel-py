from libdatachannel import (
    AACRtpPacketizer,
    OpusRtpPacketizer,
    RtpPacketizationConfig,
    RtpPacketizer,
)


def test_rtp_packetizer():
    config = RtpPacketizationConfig(
        ssrc=1234,
        cname="stream1",
        payload_type=96,
        clock_rate=48000,
        video_orientation_id=0,
    )
    packetizer = RtpPacketizer(config)
    assert packetizer.rtp_config is config
    assert packetizer.rtp_config.ssrc == 1234
    assert packetizer.rtp_config.cname == "stream1"
    # 構築できるかどうかだけ確認
    OpusRtpPacketizer(config)
    AACRtpPacketizer(config)
