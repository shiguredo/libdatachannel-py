import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from libdatachannel import (
    AV1RtpPacketizer,
    H264RtpPacketizer,
    H265RtpPacketizer,
    Message,
    NalUnit,
    OpusRtpPacketizer,
    RtpPacketizationConfig,
    RtpPacketizer,
)

# 子プロセスの起動と検証にかかる時間より十分大きい値にする
_PACKETIZER_HANG_TIMEOUT = 60


def make_rtp_config(clock_rate: int) -> RtpPacketizationConfig:
    """パケッタイザ用の RTP 設定を作る"""
    return RtpPacketizationConfig(
        ssrc=1234,
        cname="stream1",
        payload_type=96,
        clock_rate=clock_rate,
        video_orientation_id=0,
    )


def make_message(data: bytes) -> Message:
    """バイト列をそのまま持つ Message を作る"""
    message = Message(len(data))
    for index, value in enumerate(data):
        message[index] = value
    return message


def make_nal_message(nal_unit: bytes) -> Message:
    """NalUnit.Separator.Length 用に NAL 長 4 バイトを前置した Message を作る"""
    return make_message(len(nal_unit).to_bytes(4, "big") + nal_unit)


def _run_packetizer_outgoing(
    codec: str, max_fragment_size: int, input_size: int
) -> subprocess.CompletedProcess[str]:
    """恒停し得る outgoing の検証を子プロセスに分離して実行する"""
    script = Path(__file__).with_name("hang_reproduction_packetizer.py")
    return subprocess.run(
        [sys.executable, str(script), codec, str(max_fragment_size), str(input_size)],
        capture_output=True,
        text=True,
        timeout=_PACKETIZER_HANG_TIMEOUT,
        check=False,
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


@pytest.mark.timeout(10)
@pytest.mark.parametrize("max_fragment_size", [1, 2, 3], ids=["1", "2", "3"])
def test_h264_rtp_packetizer_rejects_small_max_fragment_size(max_fragment_size: int) -> None:
    """H264RtpPacketizer が小さすぎる max_fragment_size を構築時に拒否すること

    libdatachannel の generateFragments はフラグメント長から FU ヘッダ 2 バイトを引くため、
    max_fragment_size が 3 以下だと入力サイズによっては長さが 0 (offset が進まず無限ループ)
    や size_t のアンダーフロー (範囲外アクセス) になる。 構築時に拒否して outgoing に
    到達させない。
    """
    config = make_rtp_config(H264RtpPacketizer.CLOCK_RATE)
    expected = (
        "max_fragment_size must be at least 4 to fragment an H264 NAL unit, "
        f"got {max_fragment_size}"
    )
    with pytest.raises(ValueError, match=expected):
        H264RtpPacketizer(NalUnit.Separator.Length, config, max_fragment_size)


@pytest.mark.parametrize(
    ("max_fragment_size", "nal_size"),
    [(4, 5), (4, 9), (4, 1000), (65535, 131072)],
    ids=["4_5", "4_9", "4_1000", "65535_131072"],
)
def test_h264_rtp_packetizer_outgoing_with_minimum_max_fragment_size(
    max_fragment_size: int, nal_size: int
) -> None:
    """下限ちょうどの max_fragment_size で outgoing が恒停せず戻ること

    NAL 5 バイトは分割が起きる最小のサイズ (max_fragment_size + 1)、 9 バイトは
    2 * max_fragment_size + 1 で、 いずれもフラグメント長がヘッダ長をわずかに上回る
    境界である。 1000 バイトは複数に分割される正常系、 65535 はフラグメント長が
    uint16_t に切り詰められない上限で、 131072 バイトの NAL で境界を確認する。
    outgoing は引数のメッセージ列を RTP パケットに置き換えるだけで send を呼ばず、
    引数は Python 側へ書き戻されないため結果を観測できない。 恒停しないことは
    子プロセスに分離して timeout で確認する。
    """
    result = _run_packetizer_outgoing("h264", max_fragment_size, nal_size)
    assert result.returncode == 0, (
        f"h264 (max_fragment_size={max_fragment_size}, NAL {nal_size} バイト) の"
        f"outgoing が恒停した: returncode={result.returncode} stderr={result.stderr[-2000:]}"
    )


@pytest.mark.timeout(10)
@pytest.mark.parametrize("max_fragment_size", [65536, 65537], ids=["65536", "65537"])
def test_h264_rtp_packetizer_rejects_large_max_fragment_size(max_fragment_size: int) -> None:
    """H264RtpPacketizer が大きすぎる max_fragment_size を構築時に拒否すること

    generateFragments はフラグメント長を uint16_t に切り詰めるため、 65536 以上では
    切り詰めで長さが 0 や 1 になり、 小さい値を渡したときと同じ無限ループや範囲外
    アクセスが起きる。
    """
    config = make_rtp_config(H264RtpPacketizer.CLOCK_RATE)
    expected = (
        "max_fragment_size must be at most 65535 to fragment an H264 NAL unit, "
        f"got {max_fragment_size}"
    )
    with pytest.raises(ValueError, match=expected):
        H264RtpPacketizer(NalUnit.Separator.Length, config, max_fragment_size)


@pytest.mark.timeout(10)
@pytest.mark.parametrize("max_fragment_size", [1, 2, 3, 4, 5], ids=["1", "2", "3", "4", "5"])
def test_h265_rtp_packetizer_rejects_small_max_fragment_size(max_fragment_size: int) -> None:
    """H265RtpPacketizer が小さすぎる max_fragment_size を構築時に拒否すること

    H265 は FU ヘッダが 3 バイトのため、 H264 より大きい 6 が下限になる。
    """
    config = make_rtp_config(H265RtpPacketizer.CLOCK_RATE)
    expected = (
        "max_fragment_size must be at least 6 to fragment an H265 NAL unit, "
        f"got {max_fragment_size}"
    )
    with pytest.raises(ValueError, match=expected):
        H265RtpPacketizer(NalUnit.Separator.Length, config, max_fragment_size)


@pytest.mark.parametrize(
    ("max_fragment_size", "nal_size"),
    [(6, 7), (6, 13), (6, 1000), (65535, 131072)],
    ids=["6_7", "6_13", "6_1000", "65535_131072"],
)
def test_h265_rtp_packetizer_outgoing_with_minimum_max_fragment_size(
    max_fragment_size: int, nal_size: int
) -> None:
    """下限ちょうどの max_fragment_size で outgoing が恒停せず戻ること

    H265 は FU ヘッダが 3 バイトのため、 分割が起きる最小のサイズは 7 バイトになる。
    """
    result = _run_packetizer_outgoing("h265", max_fragment_size, nal_size)
    assert result.returncode == 0, (
        f"h265 (max_fragment_size={max_fragment_size}, NAL {nal_size} バイト) の"
        f"outgoing が恒停した: returncode={result.returncode} stderr={result.stderr[-2000:]}"
    )


@pytest.mark.timeout(10)
@pytest.mark.parametrize("max_fragment_size", [65536], ids=["65536"])
def test_h265_rtp_packetizer_rejects_large_max_fragment_size(max_fragment_size: int) -> None:
    """H265RtpPacketizer が大きすぎる max_fragment_size を構築時に拒否すること

    H265 も H264 と同じくフラグメント長を uint16_t に切り詰める。
    """
    config = make_rtp_config(H265RtpPacketizer.CLOCK_RATE)
    expected = (
        "max_fragment_size must be at most 65535 to fragment an H265 NAL unit, "
        f"got {max_fragment_size}"
    )
    with pytest.raises(ValueError, match=expected):
        H265RtpPacketizer(NalUnit.Separator.Length, config, max_fragment_size)


@pytest.mark.timeout(10)
@pytest.mark.parametrize("max_fragment_size", [0, 1], ids=["0", "1"])
def test_av1_rtp_packetizer_rejects_small_max_fragment_size(max_fragment_size: int) -> None:
    """AV1RtpPacketizer が小さすぎる max_fragment_size を構築時に拒否すること

    fragmentObu は payload を max_fragment_size の大きさで確保するため、 0 だと
    payload.at(0) が範囲外になり、 1 だと payloadRemaining が 0 になってループが進まない。
    """
    config = make_rtp_config(AV1RtpPacketizer.CLOCK_RATE)
    expected = (
        f"max_fragment_size must be at least 2 to fragment an AV1 OBU, got {max_fragment_size}"
    )
    with pytest.raises(ValueError, match=expected):
        AV1RtpPacketizer(AV1RtpPacketizer.Packetization.Obu, config, max_fragment_size)


def test_av1_rtp_packetizer_outgoing_with_minimum_max_fragment_size() -> None:
    """下限ちょうどの max_fragment_size (2) で outgoing が恒停せず戻ること

    この OBU は SequenceHeader ではないため、 packetizer が SequenceHeader を
    キャッシュしていない状態の確認になる。 SequenceHeader をキャッシュした状態では
    max_fragment_size が 2 + SequenceHeader 長 未満だとヒープを壊すが、 この経路は
    binding から判定できないため対象外である。
    """
    result = _run_packetizer_outgoing("av1", 2, 6)
    assert result.returncode == 0, (
        f"av1 の outgoing が恒停した: returncode={result.returncode} stderr={result.stderr[-2000:]}"
    )


@pytest.mark.timeout(10)
@pytest.mark.parametrize("codec", ["h264", "h265", "av1"], ids=["h264", "h265", "av1"])
def test_packetizer_outgoing_releases_gil(codec: str) -> None:
    """RtpPacketizer 系の outgoing() が GIL を解放して実行されること

    sys.setswitchinterval を大きくして Python 側の定期切替を止め、 GIL を待つ thread が
    呼び出し中に進行するかで判定する。 GIL を解放しない呼び出しでは、 待機 thread は
    進行できない。 映像の 3 クラスすべてで確認する。
    """
    # free-threading ビルドには GIL が無いため、 GIL 解放そのものを測れない
    # (call_guard の gil_scoped_release も no-op になる)
    if not getattr(sys, "_is_gil_enabled", lambda: True)():
        pytest.skip("GIL が無いビルド (free-threading) では GIL 解放を測れない")

    # 分割の回数を増やして呼び出し 1 回あたりの処理時間を稼ぐ
    if codec == "h264":
        config = make_rtp_config(H264RtpPacketizer.CLOCK_RATE)
        packetizer = H264RtpPacketizer(NalUnit.Separator.Length, config, 1220)
        message = make_nal_message(bytes([0x65]) + bytes(63999))
    elif codec == "h265":
        config = make_rtp_config(H265RtpPacketizer.CLOCK_RATE)
        packetizer = H265RtpPacketizer(NalUnit.Separator.Length, config, 1220)
        message = make_nal_message(bytes([0x42, 0x01]) + bytes(63998))
    else:
        config = make_rtp_config(AV1RtpPacketizer.CLOCK_RATE)
        packetizer = AV1RtpPacketizer(AV1RtpPacketizer.Packetization.Obu, config, 1220)
        message = make_message(bytes([0x32]) + bytes(63999))

    counter = 0
    stop = False

    def spin() -> None:
        nonlocal counter
        while not stop:
            counter += 1

    def call_outgoing() -> None:
        packetizer.outgoing([message], lambda out: None)

    original_interval = sys.getswitchinterval()
    thread = threading.Thread(target=spin, daemon=True)
    thread.start()
    try:
        # 待機 thread が実際に動き始めるまで待つ。 起動前に測ると、 GIL を解放しても
        # 受け取る thread がおらず進行が 0 のままになる
        deadline = time.monotonic() + 5
        while counter == 0 and time.monotonic() < deadline:
            time.sleep(0)
        assert counter > 0, "GIL を待つ thread が動き始めなかった"

        # 定期切替を止め、 GIL を解放しない限り待機 thread が動けないようにする
        sys.setswitchinterval(1.0)
        # 待機 thread に新しい switch interval で GIL を待たせ直す
        time.sleep(0)

        # GIL を解放しない呼び出し (プロパティ読み出し) の所要時間。 失敗時の診断用で、
        # 判定には使わない
        baseline_total = 0
        baseline_start = time.monotonic()
        for _ in range(200):
            baseline_total += packetizer.rtp_config.payload_type
        baseline_elapsed = time.monotonic() - baseline_start

        # GIL を解放しない限り、 待機 thread は switch interval (1 秒) のあいだ GIL を
        # 得られない。 1 秒より十分短い 50 ms のあいだ呼び続け、 その間に待機 thread が
        # 進行すれば解放されていると判定する
        released = 0
        released_start = counter
        deadline = time.monotonic() + 0.05
        while time.monotonic() < deadline:
            call_outgoing()
            released = counter - released_start
            if released:
                break
    finally:
        stop = True
        sys.setswitchinterval(original_interval)
        thread.join(timeout=5)

    assert released > 0, (
        f"outgoing() が GIL を解放しなかった (baseline {baseline_elapsed:.3f} 秒, 進行 {released})"
    )
