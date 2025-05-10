import time

from libdatachannel import WebSocket, WebSocketConfiguration


# https://github.com/paullouisageneau/libdatachannel/blob/0e40aeb058b947014a918a448ce2d346e6ab14fe/test/websocket.cpp
# を Python に直したもの
def test_websocket():
    my_message = "Hello world from libdatachannel"
    config = WebSocketConfiguration()
    config.disable_tls_verification = True
    ws = WebSocket(config)

    received = False

    def ws_on_open():
        print("WebSocket: Open")
        ws.send(my_message)

    def ws_on_error(error):
        print(f"WebSocket: Error: {error}")

    def ws_on_closed():
        print("WebSocket: Closed")

    def ws_on_message(message):
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

    ws.open("wss://echo.websocket.org:443/")

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
