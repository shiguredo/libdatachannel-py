from libdatachannel import (
    Description,
    PeerConnection,
    WebSocket,
    WebSocketConfiguration,
)


def test_data_channel_buffered_amount() -> None:
    """DataChannel の buffered_amount() が送信バッファ量を返すこと

    派生クラス側に binding が無いと MRO で Channel 側の binding が呼ばれ、 Channel が
    2 番目の基底であるために基底オフセットが加算されず、 virtual 呼び出しが誤った
    vtable スロットを読んで SIGSEGV していた。
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
