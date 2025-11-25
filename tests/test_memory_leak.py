"""
メモリリーク検出テスト

nanobind のメモリリーク検出を利用して、バインディングコードの
参照カウント問題を検出するテスト。

このテストは subprocess で実行し、終了時のリーク警告を検出する。
"""

import subprocess
import sys


def test_media_handler_chain_no_leak():
    """MediaHandler チェーンでメモリリークが発生しないことを確認"""
    code = '''
import random
import gc
from libdatachannel import (
    Configuration,
    Description,
    H264RtpPacketizer,
    NalUnit,
    OpusRtpPacketizer,
    PeerConnection,
    RtcpSrReporter,
    RtpPacketizationConfig,
)

def run():
    config = Configuration()
    config.ice_servers = []
    config.disable_auto_gathering = True
    pc = PeerConnection(config)

    audio_desc = Description.Audio("audio", Description.Direction.SendOnly)
    audio_desc.add_opus_codec(111)
    audio_track = pc.add_track(audio_desc)

    video_desc = Description.Video("video", Description.Direction.SendOnly)
    video_desc.add_h264_codec(96)
    video_track = pc.add_track(video_desc)

    video_config = RtpPacketizationConfig(
        ssrc=random.randint(1, 0xFFFFFFFF),
        cname="video-stream",
        payload_type=96,
        clock_rate=90000,
    )
    video_packetizer = H264RtpPacketizer(
        NalUnit.Separator.LongStartSequence,
        video_config,
        1200,
    )
    video_sr_reporter = RtcpSrReporter(video_config)
    video_packetizer.add_to_chain(video_sr_reporter)
    video_track.set_media_handler(video_packetizer)

    audio_config = RtpPacketizationConfig(
        ssrc=random.randint(1, 0xFFFFFFFF),
        cname="audio-stream",
        payload_type=111,
        clock_rate=48000,
    )
    audio_packetizer = OpusRtpPacketizer(audio_config)
    audio_sr_reporter = RtcpSrReporter(audio_config)
    audio_packetizer.add_to_chain(audio_sr_reporter)
    audio_track.set_media_handler(audio_packetizer)

    pc.set_local_description()

    # Track を明示的に close する
    # (pc.close() は非同期で Track を close するため、
    # プロセス終了前に完了しない可能性がある)
    video_track.close()
    audio_track.close()

    pc.close()

run()
gc.collect()
'''

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )

    # stderr に "leaked" が含まれていないことを確認
    assert "leaked" not in result.stderr.lower(), f"Memory leak detected:\n{result.stderr}"
    assert result.returncode == 0, f"Process failed:\n{result.stderr}"
