"""RTP timestamp の計算

dts (マイクロ秒) から RTP timestamp を求める。 RTP の timestamp は 32 bit のため、
初期値に経過時間を足した結果を 32 bit で wrap させる。
"""

from libdatachannel import RtpPacketizationConfig

# RTP timestamp の上限 (32 bit)
RTP_TIMESTAMP_MASK = (1 << 32) - 1

# dts はマイクロ秒
MICROSECONDS_PER_SECOND = 1_000_000


def compute_rtp_timestamp(
    start_timestamp: int,
    first_dts_usec: int,
    dts_usec: int,
    clock_rate: int,
) -> int:
    """最初の dts からの経過時間から RTP timestamp を計算する

    毎フレームの差分を足し込むと丸め誤差が累積するため、 最初の dts からの
    経過時間から計算する。 初期値 (start_timestamp) は維持したまま 32 bit で
    wrap させる。

    Args:
        start_timestamp: RTP timestamp の初期値 (乱数)
        first_dts_usec: 最初のフレームの dts (マイクロ秒)
        dts_usec: 対象のフレームの dts (マイクロ秒)
        clock_rate: クロックレート (映像は 90000、 Opus は 48000)

    Returns:
        32 bit に収めた RTP timestamp
    """
    elapsed_seconds = (dts_usec - first_dts_usec) / MICROSECONDS_PER_SECOND
    elapsed_timestamp = RtpPacketizationConfig.get_timestamp_from_seconds(
        elapsed_seconds, clock_rate
    )
    return (start_timestamp + elapsed_timestamp) & RTP_TIMESTAMP_MASK
