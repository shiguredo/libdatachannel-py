import pytest

from libdatachannel import (
    Channel,
    DataChannel,
    Description,
    PeerConnection,
    Track,
    WebSocket,
    WebSocketConfiguration,
)

# Channel の virtual メソッド。 Channel 側には binding せず、 派生クラス側で binding する
CHANNEL_VIRTUAL_METHODS: tuple[str, ...] = (
    "close",
    "send",
    "is_open",
    "is_closed",
    "max_message_size",
    "buffered_amount",
)


def test_channel_has_no_virtual_method_bindings() -> None:
    """Channel クラスに virtual メソッドの binding が無いこと

    Channel は派生クラスの 2 番目の基底であるため、 Channel 側の binding 経由で
    呼ぶと基底オフセットが加算されず、 SIGSEGV / SIGBUS になるか別の関数が実行
    されていた。 その経路が残っていないことを確認する。
    """
    for name in CHANNEL_VIRTUAL_METHODS:
        # 未定義属性として扱われ、 プロセスは落ちない
        assert not hasattr(Channel, name)


def test_derived_classes_have_virtual_method_bindings() -> None:
    """派生 3 クラスが virtual メソッドの binding を提供していること"""
    for cls in (DataChannel, Track, WebSocket):
        for name in CHANNEL_VIRTUAL_METHODS:
            assert hasattr(cls, name)


def test_data_channel_channel_methods() -> None:
    """DataChannel の is_open / is_closed / max_message_size / buffered_amount と close が動作すること

    send は未接続時の例外を確認するテストで扱う。
    """
    pc = PeerConnection()
    dc = pc.create_data_channel("channel-methods")

    # 接続前は is_open / is_closed がともに偽で、 メソッドは落ちずに値を返す
    assert not dc.is_open()
    assert not dc.is_closed()
    assert dc.max_message_size() > 0
    assert dc.buffered_amount() == 0

    # close は接続前でも動作し、 Closed 状態になる
    dc.close()
    assert dc.is_closed()


def test_track_channel_methods() -> None:
    """Track の is_open / is_closed / max_message_size / buffered_amount と close が動作すること"""
    pc = PeerConnection()
    media = Description.Video("video", Description.Direction.SendOnly)
    media.add_h264_codec(96)
    media.add_ssrc(1234, "video-send")
    track = pc.add_track(media)

    # 接続前は is_open / is_closed がともに偽で、 メソッドは落ちずに値を返す
    assert not track.is_open()
    assert not track.is_closed()
    assert track.max_message_size() > 0
    assert track.buffered_amount() == 0

    # close は接続前でも動作し、 Closed 状態になる
    track.close()
    assert track.is_closed()


def test_websocket_channel_methods() -> None:
    """WebSocket の is_open / is_closed / max_message_size / buffered_amount と close が動作すること

    WebSocket は生成直後から Closed 状態であるため、 close() では状態が変わらない。
    """
    ws = WebSocket(WebSocketConfiguration())

    # 生成直後は is_open が偽で Closed 状態、 メソッドは落ちずに値を返す
    assert not ws.is_open()
    assert ws.is_closed()
    assert ws.max_message_size() > 0
    assert ws.buffered_amount() == 0

    # 既に Closed なので close() を繰り返しても状態は変わらない
    ws.close()
    ws.close()
    assert ws.is_closed()


def test_send_with_size_on_unconnected_objects() -> None:
    """未接続では send(data, size) が RuntimeError になること

    2 引数版の binding と引数変換の経路が動作し、 クラッシュしないことを確認する。
    """
    pc = PeerConnection()
    dc = pc.create_data_channel("send-size")

    media = Description.Video("video", Description.Direction.SendOnly)
    media.add_h264_codec(96)
    media.add_ssrc(1234, "video-send")
    track = pc.add_track(media)

    ws = WebSocket(WebSocketConfiguration())

    # 未接続のオブジェクトでは送信できず RuntimeError になる
    with pytest.raises(RuntimeError):
        dc.send(b"ab", 2)
    with pytest.raises(RuntimeError):
        track.send(b"ab", 2)
    with pytest.raises(RuntimeError):
        ws.send(b"ab", 2)


def test_data_channel_buffered_amount() -> None:
    """DataChannel の buffered_amount() が送信バッファ量を返すこと

    以前は派生クラス側に binding が無く、 MRO で Channel 側の binding が呼ばれて
    いたため、 Channel が 2 番目の基底であるために基底オフセットが加算されず、
    virtual 呼び出しが誤った vtable スロットを読んで SIGSEGV していた。
    """
    pc = PeerConnection()
    dc = pc.create_data_channel("buffered-amount")

    # 未送信の状態では送信バッファは空
    assert dc.buffered_amount() == 0


def test_track_buffered_amount() -> None:
    """Track の buffered_amount() が送信バッファ量を返すこと (DataChannel と同じ経路)"""
    pc = PeerConnection()
    media = Description.Video("video", Description.Direction.SendOnly)
    media.add_h264_codec(96)
    media.add_ssrc(1234, "video-send")
    track = pc.add_track(media)

    # 未送信の状態では送信バッファは空
    assert track.buffered_amount() == 0


def test_websocket_buffered_amount() -> None:
    """WebSocket の buffered_amount() が送信バッファ量を返すこと (DataChannel と同じ経路)"""
    ws = WebSocket(WebSocketConfiguration())

    # 未接続の状態では送信バッファは空
    assert ws.buffered_amount() == 0
