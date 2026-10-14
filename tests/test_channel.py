from typing import Any

import pytest

from libdatachannel import (
    Channel,
    DataChannel,
    Description,
    FrameInfo,
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


def test_send_size_overloads_are_removed() -> None:
    """`send` の `(data, size)` 版と `send_frame` の `(data, size, info)` 版が削除されていること

    (data, size) 版は size に data の長さを超える値を渡すと範囲外を読み、 その内容を
    送信していた (SIGBUS でプロセスが落ちることもあった)。 size は len(data) から
    導出できるため size を取らない版に一本化した。
    """
    pc = PeerConnection()
    dc = pc.create_data_channel("send-size")

    media = Description.Video("video", Description.Direction.SendOnly)
    media.add_h264_codec(96)
    media.add_ssrc(1234, "video-send")
    track = pc.add_track(media)

    ws = WebSocket(WebSocketConfiguration())

    # 型検査 (ty) は削除された 2 引数版の呼び出しを引数過多として検出するため、
    # 実行時の経路は Any 経由で確認する
    dc_send: Any = dc.send
    track_send: Any = track.send
    track_send_frame: Any = track.send_frame
    ws_send: Any = ws.send

    with pytest.raises(TypeError):
        dc_send(b"ab", 2)
    with pytest.raises(TypeError):
        track_send(b"ab", 2)
    with pytest.raises(TypeError):
        track_send_frame(b"ab", 2, FrameInfo(0))
    with pytest.raises(TypeError):
        ws_send(b"ab", 2)


def test_send_without_size_on_unconnected_objects() -> None:
    """size を取らない版の send / send_frame は残り、 未接続では RuntimeError になること

    (data, size) 版の削除後も size を取らない版は従来どおり動作する。 str は bytes と
    同じく受け付ける。
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
        dc.send(b"ab")
    with pytest.raises(RuntimeError):
        dc.send("ab")
    with pytest.raises(RuntimeError):
        track.send(b"ab")
    with pytest.raises(RuntimeError):
        track.send_frame(b"ab", FrameInfo(0))
    with pytest.raises(RuntimeError):
        ws.send(b"ab")
