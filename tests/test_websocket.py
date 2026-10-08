import time

from libdatachannel import WebSocket, WebSocketConfiguration


# https://github.com/paullouisageneau/libdatachannel/blob/0e40aeb058b947014a918a448ce2d346e6ab14fe/test/websocket.cpp
# を Python に直したもの
def test_websocket(echo_websocket_server) -> None:
    my_message = "Hello world from libdatachannel"
    config = WebSocketConfiguration()
    config.disable_tls_verification = True
    ws = WebSocket(config)

    received = False

    def ws_on_open() -> None:
        print("WebSocket: Open")
        assert ws is not None
        ws.send(my_message)

    def ws_on_error(error: str) -> None:
        print(f"WebSocket: Error: {error}")

    def ws_on_closed() -> None:
        print("WebSocket: Closed")

    def ws_on_message(message: str | bytes) -> None:
        nonlocal received
        if isinstance(message, str):
            received = message == my_message
            if received:
                print("WebSocket: Received expected")
            else:
                print("WebSocket: Received UNEXPECTED message")

    ws.on_open(ws_on_open)
    ws.on_error(ws_on_error)
    ws.on_closed(ws_on_closed)
    ws.on_message(ws_on_message)

    ws.open(echo_websocket_server)

    attempts = 20
    while (not ws.is_open() or not received) and attempts > 0:
        attempts -= 1
        time.sleep(1)

    assert ws.is_open()
    assert received

    ws.close()
    time.sleep(1)

    # これが無いとリークする
    ws = None

    print("Success")


def test_websocket_send_slice(echo_websocket_server) -> None:
    """(data, size) 版の代わりにスライスした bytes を送れること

    (data, size) 版は size が data の長さを超えると範囲外を読み、 その内容を送信して
    いたため削除した。 size は len(data) から導出できるので、 前方部分送信は
    data[:size] を `send` に渡す形で等価になる。
    """
    my_message = b"0123456789"
    config = WebSocketConfiguration()
    config.disable_tls_verification = True
    ws = WebSocket(config)

    received = bytearray()
    unexpected: list[str] = []
    errors: list[str] = []

    def ws_on_open() -> None:
        assert ws is not None
        # 先頭 3 バイトだけ送る (旧 (data, size) 版の size=3 と等価)
        ws.send(my_message[:3])

    def ws_on_error(error: str) -> None:
        errors.append(error)

    def ws_on_message(message: str | bytes) -> None:
        if isinstance(message, bytes):
            received.extend(message)
        else:
            # エコーがバイナリで返らない場合はテストを失敗させる
            unexpected.append(message)

    ws.on_open(ws_on_open)
    ws.on_error(ws_on_error)
    ws.on_message(ws_on_message)

    ws.open(echo_websocket_server)

    attempts = 20
    while (not ws.is_open() or len(received) < 3) and attempts > 0:
        attempts -= 1
        time.sleep(1)

    assert not errors
    assert not unexpected
    assert received == b"012"

    ws.close()
    time.sleep(1)

    # これが無いとリークする
    ws = None
