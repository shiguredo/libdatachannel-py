import gc
import sys
import time
import weakref

import pytest

from libdatachannel import (
    Candidate,
    Configuration,
    DataChannel,
    DataChannelInit,
    Description,
    H264RtpPacketizer,
    LocalDescriptionInit,
    NalUnit,
    PeerConnection,
    PliHandler,
    Reliability,
    RtcpReceivingSession,
    RtcpSrReporter,
    RtpPacketizationConfig,
)


def test_data_channel_init():
    init = DataChannelInit()
    assert isinstance(init.reliability, Reliability)
    assert init.negotiated is False
    assert init.id is None
    assert init.protocol == ""
    init = DataChannelInit()
    init.reliability.unordered = True
    init.negotiated = True
    init.id = 7
    init.protocol = "webrtc-chat"
    assert init.reliability.unordered is True
    assert init.negotiated is True
    assert init.id == 7
    assert init.protocol == "webrtc-chat"


def test_local_description_init():
    init = LocalDescriptionInit()
    assert init.ice_ufrag is None
    assert init.ice_pwd is None
    init = LocalDescriptionInit()
    init.ice_ufrag = "abc123"
    init.ice_pwd = "xyz456"
    assert init.ice_ufrag == "abc123"
    assert init.ice_pwd == "xyz456"


def test_peerconnection_construction():
    pc = PeerConnection()
    assert pc.state() is PeerConnection.State.New
    assert pc.ice_state() is PeerConnection.IceState.New
    assert pc.gathering_state() is PeerConnection.GatheringState.New
    assert pc.signaling_state() is PeerConnection.SignalingState.Stable


def test_set_local_description():
    pc = PeerConnection()
    dc = pc.create_data_channel("chat")
    assert isinstance(dc, DataChannel)
    assert dc.label() == "chat"
    desc = pc.local_description()
    assert isinstance(desc, Description)
    assert "UDP/DTLS/SCTP" in str(desc)
    assert "sctp-port" in str(desc)
    assert "max-message-size" in str(desc)


# https://github.com/paullouisageneau/libdatachannel/blob/0e40aeb058b947014a918a448ce2d346e6ab14fe/test/track.cpp
# を Python に直したもの
def test_track():
    config1 = Configuration()
    pc1 = PeerConnection(config1)

    config2 = Configuration()
    config2.port_range_begin = 5000
    config2.port_range_end = 6000
    pc2 = PeerConnection(config2)

    def pc1_on_local_description(desc):
        print("Description 1: " + str(desc))
        pc2.set_remote_description(Description(str(desc)))

    def pc1_on_local_candidate(candidate):
        print("Candidate 1: " + str(candidate))
        pc2.add_remote_candidate(Candidate(str(candidate)))

    def pc1_on_state_change(state):
        print("State 1: " + str(state))

    def pc1_on_gathering_state_change(state):
        print("Gathering state 1: " + str(state))

    pc1.on_local_description(pc1_on_local_description)
    pc1.on_local_candidate(pc1_on_local_candidate)
    pc1.on_state_change(pc1_on_state_change)
    pc1.on_gathering_state_change(pc1_on_gathering_state_change)

    def pc2_on_local_description(desc):
        print("Description 2: " + str(desc))
        pc1.set_remote_description(Description(str(desc)))

    def pc2_on_local_candidate(candidate):
        print("Candidate 2: " + str(candidate))
        pc1.add_remote_candidate(Candidate(str(candidate)))

    def pc2_on_state_change(state):
        print("State 2: " + str(state))

    def pc2_on_gathering_state_change(state):
        print("Gathering state 2: " + str(state))

    pc2.on_local_description(pc2_on_local_description)
    pc2.on_local_candidate(pc2_on_local_candidate)
    pc2.on_state_change(pc2_on_state_change)
    pc2.on_gathering_state_change(pc2_on_gathering_state_change)

    t2 = None
    new_track_mid = ""

    def pc2_on_track(t):
        nonlocal t2
        mid = t.mid()
        print(f'Track 2: Received track with mid "{mid}"')
        if mid != new_track_mid:
            print("Wrong track mid", file=sys.stderr)
            return

        def t_on_open():
            print(f'Track 2: Track with mid "{mid}" is open')

        def t_on_closed():
            print(f'Track 2: Track with mid "{mid}" is closed')

        t.on_open(t_on_open)
        t.on_closed(t_on_closed)
        t2 = t

    pc2.on_track(pc2_on_track)

    # Test opening a track
    new_track_mid = "test"

    media = Description.Video(new_track_mid, Description.Direction.SendOnly)
    media.add_h264_codec(96)
    media.set_bitrate(3000)
    media.add_ssrc(1234, "video-send")

    media_sdp1 = str(media)
    media_sdp2 = str(Description.Media(media_sdp1))
    assert media_sdp1 == media_sdp2

    t1 = pc1.add_track(media)

    pc1.set_local_description()

    attempts = 10
    while (not t1.is_open() or t2 is None or not t2.is_open()) and attempts > 0:
        attempts -= 1
        time.sleep(1)

    assert pc1.state() == PeerConnection.State.Connected
    assert pc2.state() == PeerConnection.State.Connected

    assert t1.is_open()
    assert t2 is not None
    assert t2.is_open()

    # Test renegotiation
    new_track_mid = "added"

    media2 = Description.Video(new_track_mid, Description.Direction.SendOnly)
    media2.add_h264_codec(96)
    media2.set_bitrate(3000)
    media2.add_ssrc(2468, "video-send")

    # NOTE: Overwriting the old shared_ptr for t1 will cause it's respective
    #       track to be dropped (so it's SSRCs won't be on the description next time)
    t1 = pc1.add_track(media2)

    t2 = None
    pc1.set_local_description()

    attempts = 10
    while (not t1.is_open() or t2 is None or not t2.is_open()) and attempts > 0:
        attempts -= 1
        time.sleep(1)

    assert t1.is_open()
    assert t2 is not None
    assert t2.is_open()

    # Delay close of peer 2 to check closing works properly
    pc1.close()
    time.sleep(1)
    pc2.close()
    time.sleep(1)

    assert t1.is_closed()
    assert t2.is_closed()

    print("Success")


# test_track() と同じセットアップで、 明示的な close() なしに破棄する。 __del__
# 経由で close() が呼ばれて停止が回避されること、 ポーリング timeout 警告が出ない
# ことを recwarn で検証する。
#
# pc1 / pc2 双方の __del__ で polling timeout (最大 30 秒 × 2) を踏み得るため、
# 接続待ち 22 秒と合わせて 120 秒を上限とする。
@pytest.mark.timeout(120)
def test_destruct_without_explicit_close(recwarn):
    config1 = Configuration()
    pc1 = PeerConnection(config1)

    config2 = Configuration()
    config2.port_range_begin = 5000
    config2.port_range_end = 6000
    pc2 = PeerConnection(config2)

    # pytest stdout capture と組み合わせると issues/pending/0005 の callback I/O block
    # 経路を踏みテストが hang するため、 callback 内では print を一切行わない。
    # 根本対応は 0005 を参照。
    def pc1_on_local_description(desc):
        assert pc2 is not None
        pc2.set_remote_description(Description(str(desc)))

    def pc1_on_local_candidate(candidate):
        assert pc2 is not None
        pc2.add_remote_candidate(Candidate(str(candidate)))

    def pc1_on_state_change(state):
        pass

    def pc1_on_gathering_state_change(state):
        pass

    pc1.on_local_description(pc1_on_local_description)
    pc1.on_local_candidate(pc1_on_local_candidate)
    pc1.on_state_change(pc1_on_state_change)
    pc1.on_gathering_state_change(pc1_on_gathering_state_change)

    def pc2_on_local_description(desc):
        assert pc1 is not None
        pc1.set_remote_description(Description(str(desc)))

    def pc2_on_local_candidate(candidate):
        assert pc1 is not None
        pc1.add_remote_candidate(Candidate(str(candidate)))

    def pc2_on_state_change(state):
        pass

    def pc2_on_gathering_state_change(state):
        pass

    pc2.on_local_description(pc2_on_local_description)
    pc2.on_local_candidate(pc2_on_local_candidate)
    pc2.on_state_change(pc2_on_state_change)
    pc2.on_gathering_state_change(pc2_on_gathering_state_change)

    t2 = None
    new_track_mid = ""

    def pc2_on_track(t):
        nonlocal t2
        mid = t.mid()
        if mid != new_track_mid:
            return

        def t_on_open():
            pass

        def t_on_closed():
            pass

        t.on_open(t_on_open)
        t.on_closed(t_on_closed)
        t2 = t

    pc2.on_track(pc2_on_track)

    # Test opening a track
    new_track_mid = "test"

    media = Description.Video(new_track_mid, Description.Direction.SendOnly)
    media.add_h264_codec(96)
    media.set_bitrate(3000)
    media.add_ssrc(1234, "video-send")

    media_sdp1 = str(media)
    media_sdp2 = str(Description.Media(media_sdp1))
    assert media_sdp1 == media_sdp2

    t1 = pc1.add_track(media)

    pc1.set_local_description()

    attempts = 10
    while (not t1.is_open() or t2 is None or not t2.is_open()) and attempts > 0:
        attempts -= 1
        time.sleep(1)

    assert pc1.state() == PeerConnection.State.Connected
    assert pc2.state() == PeerConnection.State.Connected

    assert t1.is_open()
    assert t2 is not None
    assert t2.is_open()

    # callback closure による pc1/pc2 の循環参照を reset_callbacks() で明示的に
    # 断ち切る。 これで pc1 = None; pc2 = None; gc.collect() の経路で __del__ が
    # 確実に発火する。
    pc1.reset_callbacks()
    pc2.reset_callbacks()
    ref1 = weakref.ref(pc1)
    ref2 = weakref.ref(pc2)
    pc1 = None
    pc2 = None
    gc.collect()
    assert ref1() is None, "pc1 の __del__ が発火しなかった"
    assert ref2() is None, "pc2 の __del__ が発火しなかった"

    runtime_warnings = [w for w in recwarn.list if issubclass(w.category, RuntimeWarning)]
    assert not runtime_warnings, (
        f"close() の polling timeout 警告が {len(runtime_warnings)} 件発生: "
        f"{[str(w.message) for w in runtime_warnings]}"
    )


def test_del_releases_native():
    """callback 未登録の最小ケースで __del__ 経由の close を検証する。

    Free Threading 環境では refcount=0 の即時 destruct 保証が弱いので、
    gc.collect() を介して確実に発火させる。
    """
    pc = PeerConnection()
    ref = weakref.ref(pc)
    pc = None
    gc.collect()
    assert ref() is None


def test_close_is_idempotent():
    """close() を 2 回呼んでも 2 回目が早期 return で即時完了することを検証する。"""
    pc = PeerConnection()
    pc.close()
    assert pc.state() is PeerConnection.State.Closed
    start = time.monotonic()
    pc.close()
    elapsed = time.monotonic() - start
    # 2 回目は state==Closed 早期 return で即時完了する。 0.5 秒は CI ばらつきを
    # 許容しつつ 30 秒タイムアウトの regression を検出できる値。
    assert elapsed < 0.5
    assert pc.state() is PeerConnection.State.Closed


@pytest.mark.timeout(120)
def test_send_releases_gil_for_incoming_callback():
    """DataChannel.send() を連続実行している間に受信経路の PLI callback が実行されること

    映像トラックの media handler chain に PliHandler を登録し、 受信側から PLI を
    送ったあと、 送信側の DataChannel.send() を連続実行する。 送信系 binding が GIL を
    保持したまま送信経路に入ると、 PLI を処理する worker thread は GIL を取得できず
    恒久デッドロックする。 callback が DataChannel.send() の実行区間で実行された
    かどうかを送信中フラグで確認する。

    恒久デッドロックはタイミング依存で発生するため、 このテストは callback 経路の
    regression 検証であり、 修正前のコードではデッドロックして停止するか、 送信中の
    callback が観測できない。 停止した場合は main thread が GIL を保持したまま
    native lock で待つため pytest-timeout では中断できず、 外部からの kill が必要に
    なる。 接続確立と PLI の往復を見込んで 120 秒を上限とする。 なお callback を
    観測できないまま 200000 送信まで到達すると、 受信側が受信キューを消費しない
    ため送信側のメモリが数十 MB 増加し、 close() にも時間がかかる。
    """
    config1 = Configuration()
    pc1 = PeerConnection(config1)

    config2 = Configuration()
    config2.port_range_begin = 5000
    config2.port_range_end = 6000
    pc2 = PeerConnection(config2)

    # callback 内では print を行わない (pytest の stdout capture と組み合わせると
    # callback の I/O block で hang し得るため)。
    sending = False
    callback_during_send = False

    def on_pli():
        nonlocal callback_during_send
        # DataChannel.send() の実行中に callback が実行されたかどうかを記録する。
        if sending:
            callback_during_send = True

    video_ssrc = 1234
    media = Description.Video("video", Description.Direction.SendOnly)
    media.add_h264_codec(96)
    media.set_bitrate(3000)
    media.add_ssrc(video_ssrc, "video-send")
    t1 = pc1.add_track(media)

    # 送信側の media handler chain に PliHandler を登録する (whip.py と同様に
    # PliHandler を chain する構成)。 RtpPacketizationConfig の SSRC は Description
    # の SSRC と一致させる。
    rtp_config = RtpPacketizationConfig(
        video_ssrc, "send-gil-test", 96, H264RtpPacketizer.CLOCK_RATE
    )
    video_packetizer = H264RtpPacketizer(NalUnit.Separator.LongStartSequence, rtp_config, 1200)
    video_packetizer.add_to_chain(RtcpSrReporter(rtp_config))
    video_packetizer.add_to_chain(PliHandler(on_pli))
    t1.set_media_handler(video_packetizer)

    # 映像トラックとデータチャネルを同時に使う構成にする。 create_data_channel() は
    # 自動ネゴシエーションでオファーを生成するため、 先にトラックを追加しておく
    # (DataChannel を先に作るとオファーに映像の m= 行が入らない)。
    dc1 = pc1.create_data_channel("send-gil-test")

    def pc1_on_local_candidate(candidate):
        pc2.add_remote_candidate(Candidate(str(candidate)))

    def pc2_on_local_candidate(candidate):
        pc1.add_remote_candidate(Candidate(str(candidate)))

    pc1.on_local_candidate(pc1_on_local_candidate)
    pc2.on_local_candidate(pc2_on_local_candidate)

    t2 = None

    def pc2_on_track(track):
        nonlocal t2
        # 受信側のトラックから PLI を送れるようにする。 RtcpReceivingSession は
        # 受信した RTP の SSRC を PLI の送信元に使う。
        track.set_media_handler(RtcpReceivingSession())
        t2 = track

    pc2.on_track(pc2_on_track)

    # オファーは create_data_channel() 内の自動ネゴシエーションで生成されるため、
    # その通知に依存せず local_description() をポーリングして相互に SDP を交換する
    # (whip.py と同じく local_description() からオファーを取得する)。 候補が
    # 揃うまで待つ。
    offer = None
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        local = pc1.local_description()
        if local is not None:
            offer = str(local)
            if "a=end-of-candidates" in offer:
                break
        time.sleep(0.05)
    assert offer is not None, "オファーが生成されなかった"
    pc2.set_remote_description(Description(offer))

    answer = None
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        local = pc2.local_description()
        if local is not None:
            answer = str(local)
            break
        time.sleep(0.05)
    assert answer is not None, "アンサーが生成されなかった"
    pc1.set_remote_description(Description(answer))

    # 接続確立とトラック / DataChannel のオープンを待つ。
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if dc1.is_open() and t1.is_open() and t2 is not None and t2.is_open():
            break
        time.sleep(0.05)

    assert t1.is_open(), "送信側の Track が open しなかった"
    assert t2 is not None, "受信側の Track が取得できなかった"
    assert t2.is_open(), "受信側の Track が open しなかった"
    assert dc1.is_open(), "DataChannel が open しなかった"

    # SSRC の学習には RTP の受信が必要なため、 pc2 が受信するまで RTP を送る。
    nalu = b"\x00\x00\x00\x01\x65" + b"\x88" * 32
    deadline = time.monotonic() + 10
    while t2.available_amount() == 0 and time.monotonic() < deadline:
        t1.send(nalu)
        time.sleep(0.05)
    assert t2.available_amount() > 0, "受信側の Track が RTP を受信しなかった"

    # pc2 から PLI を送り、 main thread は DataChannel.send() を連続実行する。
    # 送信系 binding が GIL を解放していれば、 送信中でも PLI を処理する worker
    # thread が GIL を取得できる。 200000 送信 / 10 秒は callback を観測できなかった
    # 場合の上限で、 2000 送信ごとの PLI 再送は SSRC 学習前に落ちた PLI の
    # 取りこぼし対策。
    t2.request_keyframe()
    deadline = time.monotonic() + 10
    send_count = 0
    while not callback_during_send and time.monotonic() < deadline and send_count < 200000:
        sending = True
        dc1.send(b"ping")
        sending = False
        send_count += 1
        if send_count % 2000 == 0:
            # SSRC 学習前に落とされた PLI や、 送信中に callback が実行されなかった
            # 場合に備えて PLI を再送する。
            t2.request_keyframe()

    assert callback_during_send, (
        f"DataChannel.send() の実行中に PLI の callback が実行されなかった (send_count={send_count})"
    )

    pc1.close()
    pc2.close()


def test_config_outlives_peer_connection() -> None:
    """PeerConnection を破棄しても config() の戻り値が使えること

    config() の戻り値は PeerConnection 内部への参照のため、 親を生存させないと
    use-after-free になる。 回帰した場合はこのテストの実行中にプロセスが落ちる
    """

    pc = PeerConnection()
    config = pc.config()
    del pc
    gc.collect()
    assert isinstance(config.ice_servers, list)
