import pytest

from libdatachannel import Description, PeerConnection, PyMediaHandler, make_message


class DummyHandler(PyMediaHandler):
    def __init__(self):
        super().__init__()
        self.media_called = False
        self.incoming_called = False
        self.outgoing_called = False

    def media(self, desc):
        self.media_called = True

    def incoming(self, messages, send):
        self.incoming_called = True
        assert isinstance(messages, list)
        assert len(messages) == 1
        assert callable(send)
        send(messages[0])

    def outgoing(self, messages, send):
        self.outgoing_called = True
        assert isinstance(messages, list)
        assert len(messages) == 1
        assert callable(send)
        send(messages[0])


def test_incoming_chain():
    h = DummyHandler()

    # メッセージを 1 つ作って渡す
    msg = make_message(10)
    msgs = [msg]

    called = []

    def send_fn(m):
        called.append(m)

    h.incoming_chain(msgs, send_fn)
    assert h.incoming_called is True
    assert len(called) == 1 and called[0] is msg


def test_chaining():
    h1 = DummyHandler()
    h2 = DummyHandler()
    h1.add_to_chain(h2)

    called = []

    def send_fn(m):
        called.append(m)

    h1.incoming_chain([make_message(10)], send_fn)
    assert h1.incoming_called is True
    assert h2.incoming_called is True
    assert len(called) == 2

    h1.incoming_called = False
    h2.incoming_called = False
    called.clear()
    h1.next().incoming_chain([make_message(10)], send_fn)
    assert h1.incoming_called is False
    assert h2.incoming_called is True
    assert len(called) == 1


# cycle 検出の網羅 (自己参照 / 相互参照 / チェーン途中への接続 / 同一 handler の再追加 /
# 別チェーンでの共有) は property-based test (tests/prop_mediahandler.py) が担う。
# ここには issue の再現手順、 Track 経路、 PBT では到達できない境界値だけを置く。
def test_add_to_chain_rejects_mutual_cycle():
    """相互参照になる追加は例外になること (SEGV しないこと)"""
    h1 = DummyHandler()
    h2 = DummyHandler()
    h1.add_to_chain(h2)
    with pytest.raises(ValueError):
        h2.add_to_chain(h1)
    # 壊れた chain が残っていないことを確認する
    assert h1.next() is h2
    assert h2.next() is None
    assert h1.last() is h2


def test_chain_media_handler_rejects_cycle():
    """Track.chain_media_handler でも cycle が検出されること"""
    pc = PeerConnection()
    media = Description.Video("video", Description.Direction.SendOnly)
    media.add_h264_codec(96)
    media.add_ssrc(1234, "video-send")
    track = pc.add_track(media)

    h1 = DummyHandler()
    h2 = DummyHandler()
    track.set_media_handler(h1)
    track.chain_media_handler(h2)
    # h1 -> h2 の chain に h1 を繋ごうとすると cycle になる
    with pytest.raises(ValueError):
        track.chain_media_handler(h1)
    assert h1.last() is h2


def test_add_to_chain_accepts_chain_at_limit():
    """上限ちょうどの長さのチェーンへは接続できること"""
    handlers = [DummyHandler() for _ in range(1024 + 1)]
    for i in range(1023):
        handlers[i].set_next(handlers[i + 1])
    # 1024 ノードのチェーンの末尾へ繋ぐ
    handlers[0].add_to_chain(handlers[1024])
    assert handlers[0].last() is handlers[1024]


def test_add_to_chain_rejects_too_long_chain():
    """上限を超える長さのチェーンへの接続は上限エラーで拒否されること"""
    handlers = [DummyHandler() for _ in range(1026 + 1)]
    for i in range(1025):
        handlers[i].set_next(handlers[i + 1])
    fresh = DummyHandler()
    with pytest.raises(ValueError, match="did not reach its end within"):
        handlers[0].add_to_chain(fresh)
