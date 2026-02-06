"""Free-Threading 環境での並列アクセステスト

このテストは Python 3.13t/3.14t の Free-Threading ビルドでのみ実行される。
GIL ビルドではスキップされる。
"""

import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from libdatachannel import (
    Candidate,
    Configuration,
    DataChannelInit,
    Description,
    FrameInfo,
    IceServer,
    Message,
    NalUnit,
    PeerConnection,
    Reliability,
    RtpPacketizationConfig,
    WebSocket,
    WebSocketConfiguration,
    WebSocketServer,
    WebSocketServerConfiguration,
)

# Free-Threading ビルドかどうかを確認
is_free_threading = hasattr(sys, "_is_gil_enabled") and not sys._is_gil_enabled()

requires_free_threading = pytest.mark.skipif(
    not is_free_threading, reason="Free-Threading build required"
)


@requires_free_threading
def test_gil_status():
    """GIL が無効化されていることを確認"""
    assert hasattr(sys, "_is_gil_enabled")
    assert not sys._is_gil_enabled(), "GIL should be disabled in Free-Threading build"


@requires_free_threading
def test_concurrent_configuration_creation():
    """複数スレッドから同時に Configuration を生成"""
    results = {}
    lock = threading.Lock()
    barrier = threading.Barrier(4)

    def create_configs(thread_id: int):
        barrier.wait()
        configs = []
        for i in range(100):
            config = Configuration()
            config.mtu = 1200 + thread_id * 100 + i
            configs.append(config)
        with lock:
            results[thread_id] = configs

    threads = []
    for i in range(4):
        t = threading.Thread(target=create_configs, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    assert len(results) == 4
    for thread_id, configs in results.items():
        assert len(configs) == 100
        for i, config in enumerate(configs):
            assert config.mtu == 1200 + thread_id * 100 + i


@requires_free_threading
def test_concurrent_candidate_parsing():
    """複数スレッドから同時に Candidate をパース"""
    candidate_str = "candidate:1 1 UDP 2122194687 192.168.1.1 12345 typ host"
    errors = []
    results = {}
    lock = threading.Lock()
    barrier = threading.Barrier(4)

    def parse_candidates(thread_id: int):
        barrier.wait()
        candidates = []
        for _ in range(100):
            try:
                c = Candidate(candidate_str)
                candidates.append(c)
            except Exception as e:
                with lock:
                    errors.append(e)
        with lock:
            results[thread_id] = candidates

    threads = []
    for i in range(4):
        t = threading.Thread(target=parse_candidates, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    assert len(errors) == 0, f"Errors: {errors}"
    assert len(results) == 4
    for thread_id, candidates in results.items():
        assert len(candidates) == 100


@requires_free_threading
def test_concurrent_description_creation():
    """複数スレッドから同時に Description を生成・操作"""
    errors = []
    lock = threading.Lock()
    barrier = threading.Barrier(4)

    def create_descriptions(thread_id: int):
        barrier.wait()
        for i in range(50):
            try:
                video = Description.Video(
                    f"video-{thread_id}-{i}", Description.Direction.SendOnly
                )
                video.add_h264_codec(96)
                video.set_bitrate(3000)
                video.add_ssrc(1000 + thread_id * 100 + i, "video-send")
                sdp = str(video)
                # パースし直してラウンドトリップを確認
                reparsed = Description.Media(sdp)
                assert str(reparsed) == sdp
            except Exception as e:
                with lock:
                    errors.append(e)

    threads = []
    for i in range(4):
        t = threading.Thread(target=create_descriptions, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    assert len(errors) == 0, f"Errors: {errors}"


@requires_free_threading
def test_concurrent_message_creation():
    """複数スレッドから同時に Message を生成・操作"""
    results = {}
    lock = threading.Lock()
    barrier = threading.Barrier(4)

    def create_messages(thread_id: int):
        barrier.wait()
        messages = []
        for i in range(100):
            msg = Message(64, Message.Type.Binary)
            msg[0] = thread_id
            msg[1] = i % 256
            messages.append(msg)
        with lock:
            results[thread_id] = messages

    threads = []
    for i in range(4):
        t = threading.Thread(target=create_messages, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    assert len(results) == 4
    for thread_id, messages in results.items():
        assert len(messages) == 100
        for i, msg in enumerate(messages):
            assert msg[0] == thread_id
            assert msg[1] == i % 256


@requires_free_threading
def test_concurrent_peerconnection_creation():
    """複数スレッドから同時に PeerConnection を生成"""
    results = {}
    lock = threading.Lock()
    barrier = threading.Barrier(4)

    def create_pcs(thread_id: int):
        barrier.wait()
        pcs = []
        for _ in range(10):
            config = Configuration()
            pc = PeerConnection(config)
            assert pc.state() is PeerConnection.State.New
            pcs.append(pc)
        # クリーンアップ
        for pc in pcs:
            pc.close()
        with lock:
            results[thread_id] = len(pcs)

    threads = []
    for i in range(4):
        t = threading.Thread(target=create_pcs, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    assert len(results) == 4
    for thread_id, count in results.items():
        assert count == 10


@requires_free_threading
def test_concurrent_peerconnection_callback_registration():
    """複数スレッドから同時に PeerConnection のコールバックを登録"""
    pc = PeerConnection()
    barrier = threading.Barrier(4)
    registration_count = [0]
    lock = threading.Lock()

    def register_callbacks(thread_id: int):
        barrier.wait()
        for i in range(100):
            def on_state(state, tid=thread_id, idx=i):
                pass

            def on_ice_state(state, tid=thread_id, idx=i):
                pass

            def on_gathering(state, tid=thread_id, idx=i):
                pass

            pc.on_state_change(on_state)
            pc.on_ice_state_change(on_ice_state)
            pc.on_gathering_state_change(on_gathering)

            with lock:
                registration_count[0] += 1

    threads = []
    for i in range(4):
        t = threading.Thread(target=register_callbacks, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    assert registration_count[0] == 400
    pc.close()


@requires_free_threading
def test_concurrent_datachannel_creation():
    """複数スレッドから同時に DataChannel を生成"""
    pc = PeerConnection()
    results = {}
    lock = threading.Lock()
    barrier = threading.Barrier(4)

    def create_channels(thread_id: int):
        barrier.wait()
        channels = []
        for i in range(10):
            label = f"ch-{thread_id}-{i}"
            dc = pc.create_data_channel(label)
            channels.append(dc)
        with lock:
            results[thread_id] = channels

    threads = []
    for i in range(4):
        t = threading.Thread(target=create_channels, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    assert len(results) == 4
    for thread_id, channels in results.items():
        assert len(channels) == 10
        for i, dc in enumerate(channels):
            assert dc.label() == f"ch-{thread_id}-{i}"

    pc.close()


@requires_free_threading
def test_concurrent_rtp_config_creation():
    """複数スレッドから同時に RtpPacketizationConfig を生成"""
    results = {}
    lock = threading.Lock()
    barrier = threading.Barrier(4)

    def create_configs(thread_id: int):
        barrier.wait()
        configs = []
        for i in range(100):
            ssrc = thread_id * 10000 + i
            config = RtpPacketizationConfig(ssrc, f"cname-{thread_id}", 96, 90000)
            assert config.ssrc == ssrc
            configs.append(config)
        with lock:
            results[thread_id] = configs

    threads = []
    for i in range(4):
        t = threading.Thread(target=create_configs, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    assert len(results) == 4
    for thread_id, configs in results.items():
        assert len(configs) == 100


@requires_free_threading
def test_concurrent_nalunit_operations():
    """複数スレッドから同時に NalUnit を操作"""
    errors = []
    lock = threading.Lock()
    barrier = threading.Barrier(4)

    def operate_nalunits(thread_id: int):
        barrier.wait()
        for i in range(50):
            try:
                nalu = NalUnit(128, True, NalUnit.Type.H264)
                nalu.set_unit_type(5)
                assert nalu.unit_type() == 5
                nalu.set_forbidden_bit(False)
                assert nalu.forbidden_bit() is False
            except Exception as e:
                with lock:
                    errors.append(e)

    threads = []
    for i in range(4):
        t = threading.Thread(target=operate_nalunits, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    assert len(errors) == 0, f"Errors: {errors}"


@requires_free_threading
def test_multiple_peerconnections_parallel():
    """複数の PeerConnection インスタンスを並列で操作"""
    results = {}
    lock = threading.Lock()

    def create_and_use_pc(pc_id: int):
        config = Configuration()
        pc = PeerConnection(config)

        dc = pc.create_data_channel(f"channel-{pc_id}")
        assert dc.label() == f"channel-{pc_id}"

        desc = pc.local_description()
        assert desc is not None
        assert "UDP/DTLS/SCTP" in str(desc)

        pc.close()

        with lock:
            results[pc_id] = True

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(create_and_use_pc, i) for i in range(4)]
        for f in futures:
            f.result()

    assert len(results) == 4
    for pc_id, success in results.items():
        assert success


@requires_free_threading
def test_concurrent_ice_server_creation():
    """複数スレッドから同時に IceServer を生成"""
    results = {}
    lock = threading.Lock()
    barrier = threading.Barrier(4)

    def create_servers(thread_id: int):
        barrier.wait()
        servers = []
        for i in range(100):
            server = IceServer(f"stun:stun{thread_id}.example.com:{3478 + i}")
            servers.append(server)
        with lock:
            results[thread_id] = servers

    threads = []
    for i in range(4):
        t = threading.Thread(target=create_servers, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    assert len(results) == 4
    for thread_id, servers in results.items():
        assert len(servers) == 100


# GIL ビルドでも実行されるテスト（Free-Threading 環境の検出テスト）
def test_free_threading_detection():
    """Free-Threading 環境の検出機能をテスト"""
    is_gil_enabled = getattr(sys, "_is_gil_enabled", None)
    if is_gil_enabled is not None:
        gil_enabled = is_gil_enabled()
        # Python 3.13+ では _is_gil_enabled が存在する
        assert isinstance(gil_enabled, bool)
    # Python 3.12 以下では _is_gil_enabled が存在しない
