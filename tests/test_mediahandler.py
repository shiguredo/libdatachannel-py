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


def test_add_to_chain_rejects_self_cycle():
    """自分自身を chain に追加しようとすると例外になること (SEGV しないこと)"""
    h = DummyHandler()
    with pytest.raises(ValueError):
        h.add_to_chain(h)
    # 例外になった後も chain は壊れていない
    assert h.next() is None


def test_set_next_rejects_self_cycle():
    """自分自身を next に設定しようとすると例外になること (SEGV しないこと)"""
    h = DummyHandler()
    with pytest.raises(ValueError):
        h.set_next(h)
    assert h.next() is None


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


def test_set_next_rejects_cycle_into_own_chain():
    """自分の chain の先頭へ戻る next は例外になること"""
    h1 = DummyHandler()
    h2 = DummyHandler()
    h1.set_next(h2)
    with pytest.raises(ValueError):
        h2.set_next(h1)
    assert h2.next() is None


def test_add_to_chain_rejects_cycle_into_own_chain():
    """自分の chain の途中へ戻る追加は例外になること"""
    h1 = DummyHandler()
    h2 = DummyHandler()
    h3 = DummyHandler()
    h1.add_to_chain(h2)
    with pytest.raises(ValueError):
        h2.add_to_chain(h1)
    # h3 は無関係なので接続できる
    h2.add_to_chain(h3)
    assert h2.next() is h3
    assert h2.last() is h3


def test_add_to_chain_allows_shared_handler_in_other_chain():
    """別の chain で既に使われている handler を末尾に追加できること (cycle ではない)"""
    h1 = DummyHandler()
    h2 = DummyHandler()
    h3 = DummyHandler()
    h1.add_to_chain(h3)
    h2.add_to_chain(h3)
    assert h1.next() is h3
    assert h2.next() is h3


def test_add_to_chain_rejects_same_handler_twice():
    """同じ handler を 2 回追加しようとすると例外になること

    chain の末尾へ同じ handler を再度連結すると自己 cycle になる。
    """
    h1 = DummyHandler()
    h2 = DummyHandler()
    h3 = DummyHandler()
    h1.add_to_chain(h2)
    h1.add_to_chain(h3)
    with pytest.raises(ValueError):
        h1.add_to_chain(h3)
    assert h1.last() is h3


def test_add_to_chain_rejects_handler_in_middle_of_own_chain():
    """自分の chain の途中の handler を末尾へ繋ごうとすると例外になること"""
    h1 = DummyHandler()
    h2 = DummyHandler()
    h3 = DummyHandler()
    h1.set_next(h2)
    h2.set_next(h3)
    with pytest.raises(ValueError):
        h1.add_to_chain(h2)
    assert h1.last() is h3


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
