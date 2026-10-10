import gc
import sys
import threading
import time
import weakref
from collections.abc import Callable

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
    Track,
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

    # Track の open / close はポーリングせず、 callback の通知をイベントで待つ。
    # callback が呼ばれなければ wait() が timeout で False を返し、 assert で失敗する。
    t1_opened = threading.Event()
    t1_closed = threading.Event()
    t2_opened = threading.Event()
    t2_closed = threading.Event()

    def pc2_on_track(t):
        nonlocal t2
        mid = t.mid()
        print(f'Track 2: Received track with mid "{mid}"')
        if mid != new_track_mid:
            print("Wrong track mid", file=sys.stderr)
            return

        def t_on_open():
            print(f'Track 2: Track with mid "{mid}" is open')
            t2_opened.set()

        def t_on_closed():
            print(f'Track 2: Track with mid "{mid}" is closed')
            t2_closed.set()

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
    t1.on_open(t1_opened.set)
    t1.on_closed(t1_closed.set)

    pc1.set_local_description()

    # callback から通知されるまで待つ (ポーリングしない)。 旧実装は 1 秒 × 10 回の
    # ポーリングだったため、 待ち時間の上限はそれより余裕を持たせた 20 秒とする。
    assert t1_opened.wait(timeout=20), "送信側の Track が open しなかった"
    assert t2_opened.wait(timeout=20), "受信側の Track が open しなかった"

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

    # t2 側の callback は外側の Event をセル参照で捕捉するため、 再ネゴシエーションで
    # イベントを作り直すと 1 本目の Track の通知でも新しい Event が set される。
    # 偽陽性を避けるため、 1 本目の open は直前の assert で確認し、 最後に
    # t2.is_closed() で閉じていることを担保する。
    t1_opened = threading.Event()
    t1_closed = threading.Event()
    t2_opened = threading.Event()
    t2_closed = threading.Event()

    t1 = pc1.add_track(media2)
    t1.on_open(t1_opened.set)
    t1.on_closed(t1_closed.set)

    t2 = None
    pc1.set_local_description()

    assert t1_opened.wait(timeout=20), "再ネゴシエーション後の送信側の Track が open しなかった"
    assert t2_opened.wait(timeout=20), "再ネゴシエーション後の受信側の Track が open しなかった"

    assert t1.is_open()
    assert t2 is not None
    assert t2.is_open()

    # pc2 の close を遅らせて、 先に pc1 を閉じても正常に閉じられることを確認する。
    # 遅延は固定時間ではなく、 pc1 側の Track が閉じた通知 (on_closed) で待つ。
    pc1.close()
    assert t1_closed.wait(timeout=20), "送信側の Track が閉じなかった"
    pc2.close()
    assert t2_closed.wait(timeout=20), "受信側の Track が閉じなかった"

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

    # このテストは callback 経由の恒停 (destructor 経路の同期 callback) が既知で、
    # 実行して検証できない。 イベント待ちに置き換えると未検証の差分になるため、
    # 接続確立待ちは既存のポーリングのまま残す。 根本対応時にまとめて直す。
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


def make_loopback_with_pli(
    on_pli: Callable[[], None],
) -> tuple[PeerConnection, PeerConnection, Track, Track, DataChannel]:
    """映像トラックに PliHandler を chain したループバック接続を作る

    送信側 (pc1) で映像トラックと DataChannel を開き、 受信側 (pc2) のトラックには
    RtcpReceivingSession を設定して PLI を送れるようにする。 受信した PLI は送信側の
    PliHandler の callback (on_pli) が worker thread で処理する。 返り値は
    (pc1, pc2, t1, t2, dc1)。
    """
    config1 = Configuration()
    pc1 = PeerConnection(config1)

    config2 = Configuration()
    config2.port_range_begin = 5000
    config2.port_range_end = 6000
    pc2 = PeerConnection(config2)

    video_ssrc = 1234
    media = Description.Video("video", Description.Direction.SendOnly)
    media.add_h264_codec(96)
    media.set_bitrate(3000)
    media.add_ssrc(video_ssrc, "video-send")
    t1 = pc1.add_track(media)

    # 接続完了は callback の通知をイベントで待つ (ポーリングしない)。 callback が
    # 呼ばれなければ wait() が timeout で False を返し、 assert で失敗する。
    t1_opened = threading.Event()
    t2_opened = threading.Event()
    dc1_opened = threading.Event()
    gathering_completed = threading.Event()

    # ICE 候補の収集完了を通知する。 libdatachannel は収集完了時に
    # endLocalCandidates() で SDP に a=end-of-candidates を入れてから gathering
    # state を Complete に変えるため、 この通知が来た時点の local_description() には
    # a=end-of-candidates が含まれる。
    def pc1_on_gathering_state_change(state) -> None:
        if state == PeerConnection.GatheringState.Complete:
            gathering_completed.set()

    pc1.on_gathering_state_change(pc1_on_gathering_state_change)
    t1.on_open(t1_opened.set)

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
    dc1.on_open(dc1_opened.set)

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
        track.on_open(t2_opened.set)
        t2 = track

    pc2.on_track(pc2_on_track)

    # pc2 側のアンサー生成も callback の通知をイベントで待つ。
    answer_generated = threading.Event()

    def pc2_on_local_description(desc) -> None:
        answer_generated.set()

    pc2.on_local_description(pc2_on_local_description)

    # オファーは create_data_channel() 内の自動ネゴシエーションで生成されるため、
    # その通知に依存せず、 ICE 候補の収集完了を待ってから local_description() で
    # オファーを取得する (whip.py と同じく local_description() から取得する)。
    assert gathering_completed.wait(timeout=20), "ICE 候補の収集が完了しなかった"
    local = pc1.local_description()
    assert local is not None, "オファーが生成されなかった"
    offer = str(local)
    assert "a=end-of-candidates" in offer, "オファーに a=end-of-candidates が含まれなかった"
    pc2.set_remote_description(Description(offer))

    # アンサーは pc2 の on_local_description の通知で待つ。
    assert answer_generated.wait(timeout=20), "アンサーが生成されなかった"
    local = pc2.local_description()
    assert local is not None, "アンサーが生成されなかった"
    answer = str(local)
    pc1.set_remote_description(Description(answer))

    # 接続確立とトラック / DataChannel のオープンを callback で待つ。
    assert t1_opened.wait(timeout=20), "送信側の Track が open しなかった"
    assert t2_opened.wait(timeout=20), "受信側の Track が open しなかった"
    assert dc1_opened.wait(timeout=20), "DataChannel が open しなかった"

    assert t1.is_open(), "送信側の Track が open しなかった"
    assert t2 is not None, "受信側の Track が取得できなかった"
    assert t2.is_open(), "受信側の Track が open しなかった"
    assert dc1.is_open(), "DataChannel が open しなかった"

    # SSRC の学習には RTP の受信が必要なため、 pc2 が受信するまで RTP を送る。
    # 受信側の Track は media handler (RtcpReceivingSession) が受信フレームを消費し、
    # 受信を通知する callback が無いため、 available_amount() を条件に送信を繰り返す
    # 必要がある。 この待ちは「次の再送までの間隔」であり、 イベント待ちにはできない。
    nalu = b"\x00\x00\x00\x01\x65" + b"\x88" * 32
    deadline = time.monotonic() + 10
    while t2.available_amount() == 0 and time.monotonic() < deadline:
        t1.send(nalu)
        time.sleep(0.05)
    assert t2.available_amount() > 0, "受信側の Track が RTP を受信しなかった"

    return pc1, pc2, t1, t2, dc1


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

    # callback 内では print を行わない (pytest の stdout capture と組み合わせると
    # callback の I/O block で hang し得るため)。
    sending = False
    callback_during_send = False

    def on_pli() -> None:
        nonlocal callback_during_send
        # DataChannel.send() の実行中に callback が実行されたかどうかを記録する。
        if sending:
            callback_during_send = True

    pc1, pc2, _t1, t2, dc1 = make_loopback_with_pli(on_pli)

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


@pytest.mark.timeout(120)
def test_request_keyframe_releases_gil_for_incoming_callback() -> None:
    """Track.request_keyframe() が送信経路を通り PLI callback が動くこと

    RtcpReceivingSession が send callback (= impl()->transportSend) を呼ぶことで送信
    経路に入る。 戻り値が真であることが送信経路を通った証拠になる。 送った PLI は
    pc1 の PliHandler が受信し、 worker thread が Python callback を実行する (実行
    区間中かどうかは送信中フラグで見るが、 通常の GIL 切替でも成立し得るため GIL 解放の
    証拠にはならない。 GIL 解放そのものは test_request_media_control_releases_gil で
    測る)。 request_bitrate() は PLI ではなく REMB を送るため、 このテストの対象外。

    接続確立と PLI の往復を見込んで 120 秒を上限とする。
    """
    sending = False
    callback_during_send = False

    def on_pli() -> None:
        nonlocal callback_during_send
        # request_keyframe() の実行中に callback が動いたか記録する。
        if sending:
            callback_during_send = True

    pc1, pc2, _t1, t2, _dc1 = make_loopback_with_pli(on_pli)

    deadline = time.monotonic() + 10
    call_count = 0
    while not callback_during_send and time.monotonic() < deadline and call_count < 20000:
        sending = True
        sent = t2.request_keyframe()
        sending = False
        assert sent, "RtcpReceivingSession.request_keyframe() が PLI を送らなかった"
        call_count += 1

    assert callback_during_send, (
        f"Track.request_keyframe() の実行中に PLI の callback が実行されなかった (call_count={call_count})"
    )

    pc1.close()
    pc2.close()


@pytest.mark.parametrize(
    "operation",
    ["request_keyframe", "request_bitrate"],
    ids=["request_keyframe", "request_bitrate"],
)
@pytest.mark.timeout(120)
def test_request_media_control_releases_gil(operation: str) -> None:
    """Track の request_keyframe() / request_bitrate() が GIL を解放して実行されること

    sys.setswitchinterval を大きくして Python 側の定期切替を止め、 GIL を待つ thread が
    呼び出し中に進行するかで判定する。 GIL を解放しない呼び出しでは、 待機 thread は
    進行できない。 呼び出しは送信経路を通るため RtcpReceivingSession を持つ受信側
    トラックを使う (戻り値が真であることが送信経路を通った証拠)。
    """
    # free-threading ビルドには GIL が無いため、 GIL 解放そのものを測れない
    # (call_guard の gil_scoped_release も no-op になる)。
    if not getattr(sys, "_is_gil_enabled", lambda: True)():
        pytest.skip("GIL が無いビルド (free-threading) では GIL 解放を測れない")

    pc1, pc2, _t1, t2, _dc1 = make_loopback_with_pli(lambda: None)

    def call_media_control() -> bool:
        if operation == "request_keyframe":
            return t2.request_keyframe()
        return t2.request_bitrate(500000)

    counter = 0
    stop = False

    def spin() -> None:
        nonlocal counter
        while not stop:
            counter += 1

    original_interval = sys.getswitchinterval()
    thread = threading.Thread(target=spin, daemon=True)
    thread.start()
    try:
        # 待機 thread が実際に動き始めるまで待つ。 起動前に測ると、 GIL を解放しても
        # 受け取る thread がおらず進行が 0 のままになる (CI で観測された flaky)。
        deadline = time.monotonic() + 5
        while counter == 0 and time.monotonic() < deadline:
            time.sleep(0)
        assert counter > 0, "GIL を待つ thread が動き始めなかった"

        # 定期切替を止め、 GIL を解放しない限り待機 thread が動けないようにする
        sys.setswitchinterval(1.0)
        # 待機 thread に新しい switch interval で GIL を待たせ直す。 ここで一度 GIL を
        # 手放して保留中の受け渡しを解消する。 これをしないと、 待機 thread は変更前の
        # 短い interval (既定 5 ms) で待ち続けているため、 計測中に周期的な受け渡しが
        # 起きて、 GIL を解放しない呼び出しでも進行が観測されてしまう
        time.sleep(0)
        # GIL を解放しない呼び出し (description()) は待機 thread に GIL を渡さない。
        # この baseline は失敗時の診断用で、 判定には使わない (待機 thread の待ち直しが
        # 効かない環境では baseline 中にも受け渡しが起き得るため)
        baseline_start = time.monotonic()
        for _ in range(200):
            t2.description()
        baseline_elapsed = time.monotonic() - baseline_start

        # GIL を解放しない限り、 待機 thread は switch interval (1 秒) のあいだ GIL を
        # 得られない。 1 秒より十分短い 50 ms のあいだ呼び続け、 その間に待機 thread が
        # 進行すれば解放されていると判定する。 50 ms では周期的な受け渡しが起きないため、
        # 進行があれば解放によるものだと断定できる。 (1 回だけの計測は、 解放窓が µs の
        # ときに待機 thread がその窓で走り出せず偽陰性になる)
        released = 0
        released_start = counter
        deadline = time.monotonic() + 0.05
        while time.monotonic() < deadline:
            assert call_media_control(), f"{operation}() が送信経路を通らなかった"
            released = counter - released_start
            if released:
                break
    finally:
        stop = True
        sys.setswitchinterval(original_interval)
        thread.join(timeout=10)

    assert released > 0, (
        f"{operation}() が GIL を解放しなかった "
        f"(released={released}, baseline_elapsed={baseline_elapsed:.6f})"
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


def test_data_channel_close_releases_gil() -> None:
    """DataChannel.close() が GIL を解放して実行されること

    close() は SctpTransport::closeStream() で送信経路と同じ mutex を取るため、 GIL を
    保持したまま待つと、 送信経路の Python callback が GIL を取れずに循環待ちになる。
    close() の呼び出し中に GIL を待つ thread が進行するかで判定する。
    """
    # free-threading ビルドには GIL が無いため、 GIL 解放そのものを測れない
    if not getattr(sys, "_is_gil_enabled", lambda: True)():
        pytest.skip("GIL が無いビルド (free-threading) では GIL 解放を測れない")

    _pc1, _pc2, _t1, _t2, data_channel = make_loopback_with_pli(lambda: None)

    counter = 0
    stop = False

    def spin() -> None:
        nonlocal counter
        while not stop:
            counter += 1

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
        # 待機 thread に新しい switch interval で GIL を待たせ直す (上のコメント参照)
        time.sleep(0)

        # close() は呼び出し全体で GIL を解放するが、 呼び出し自体が短いため 1 回の
        # 計測では待機 thread が動き出せず偽陰性になり得る。 50 ms のあいだ呼び続け、
        # その間に進行すれば解放されていると判定する。
        # 2 回目以降の close() は内部では no-op になるが、 call_guard は毎回通る
        released = 0
        released_start = counter
        deadline = time.monotonic() + 0.05
        while time.monotonic() < deadline:
            data_channel.close()
            released = counter - released_start
            if released:
                break
    finally:
        stop = True
        sys.setswitchinterval(original_interval)
        thread.join(timeout=5)
        # ループバックは相互参照を持つため、 明示的に回収する。 閉じた DataChannel が
        # 残ると nanobind のリーク警告がインタプリタ終了時に出る
        del data_channel
        del _pc1, _pc2, _t1, _t2
        gc.collect()

    assert released > 0, "close() が GIL を解放していない"
