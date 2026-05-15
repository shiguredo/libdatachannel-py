import time

from libdatachannel import (
    WebSocket,
    WebSocketConfiguration,
    WebSocketServer,
    WebSocketServerConfiguration,
)


# https://github.com/paullouisageneau/libdatachannel/blob/0e40aeb058b947014a918a448ce2d346e6ab14fe/test/websocketserver.cpp#L1
# を Python に直したもの
def test_websocketserver():
    server_config = WebSocketServerConfiguration()
    server_config.port = 48080
    server_config.enable_tls = True
    server_config.bind_address = "127.0.0.1"
    server_config.max_message_size = 1000
    server = WebSocketServer(server_config)

    client = None

    def server_on_client(incoming):
        nonlocal client
        print("WebSocketServer: Client connection received")
        client = incoming

        addr = client.remote_address()
        if addr is not None:
            print(f"WebSocketServer: Client remote address is {addr}")

        def client_on_open():
            nonlocal client
            print("WebSocketServer: Client connection open")
            path = client.path()
            if path is not None:
                print(f"WebSocketServer: Requested path is {path}")

        def client_on_closed():
            print("WebSocketServer: Client connection closed")

        def client_on_message(message):
            nonlocal client
            client.send(message)

        client.on_open(client_on_open)
        client.on_closed(client_on_closed)
        client.on_message(client_on_message)

    server.on_client(server_on_client)

    config = WebSocketConfiguration()
    config.disable_tls_verification = True
    ws = WebSocket(config)

    my_message = "Hello world from client"

    def ws_on_open():
        print("WebSocket: Open")
        ws.send(b"\x00" * 1001)
        ws.send(my_message)

    def ws_on_closed():
        print("WebSocket: Closed")

    ws.on_open(ws_on_open)
    ws.on_closed(ws_on_closed)

    received = False
    max_size_received = False

    def ws_on_message(message):
        nonlocal received
        nonlocal max_size_received
        if isinstance(message, str):
            received = message == my_message
            if received:
                print("WebSocket: Received expected message")
            else:
                print("WebSocket: Received UNEXPECTED message")
        else:
            max_size_received = len(message) == 1000
            if max_size_received:
                print("WebSocket: Received large message truncated at max size")
            else:
                print("WebSocket: Received large message NOT TRUNCATED")

    ws.on_message(ws_on_message)

    ws.open("wss://localhost:48080/")

    attempts = 15
    while (not ws.is_open() or not received) and attempts > 0:
        attempts -= 1
        time.sleep(1)

    assert ws.is_open()
    assert max_size_received
    assert received

    ws.close()
    time.sleep(1)

    server.stop()
    time.sleep(1)

    # これが無いとリークする
    ws = None
    server = None
    client = None

    print("Success")


# close()/stop() を呼ばずに WebSocket/WebSocketServer の参照を切っても
# destructor で hang しないことを確認する。
# PeerConnection と異なり、 WebSocket は destructor で blocking しない
# (impl::WebSocket::~WebSocket は PLOG のみ)。 WebSocketServer は stop() で
# mThread.join() するが、 tcpServer->close() により accept loop が即時抜ける
# ため、 GIL 保持下でも join() は短時間で完了する。 したがって PeerConnection
# のような __del__ で事前 close を呼ぶ wrapper は不要。
def test_destruct_without_explicit_close():
    server_config = WebSocketServerConfiguration()
    server_config.port = 48081
    server_config.enable_tls = True
    server_config.bind_address = "127.0.0.1"
    server_config.max_message_size = 1000
    server = WebSocketServer(server_config)

    client = None

    def server_on_client(incoming):
        nonlocal client
        client = incoming

        def client_on_open():
            nonlocal client
            print("WebSocketServer: Client connection open")

        def client_on_closed():
            print("WebSocketServer: Client connection closed")

        def client_on_message(message):
            nonlocal client
            client.send(message)

        client.on_open(client_on_open)
        client.on_closed(client_on_closed)
        client.on_message(client_on_message)

    server.on_client(server_on_client)

    config = WebSocketConfiguration()
    config.disable_tls_verification = True
    ws = WebSocket(config)

    my_message = "Hello world from client"

    def ws_on_open():
        print("WebSocket: Open")
        ws.send(my_message)

    def ws_on_closed():
        print("WebSocket: Closed")

    ws.on_open(ws_on_open)
    ws.on_closed(ws_on_closed)

    received = False

    def ws_on_message(message):
        nonlocal received
        if isinstance(message, str):
            received = message == my_message

    ws.on_message(ws_on_message)

    ws.open("wss://localhost:48081/")

    attempts = 15
    while (not ws.is_open() or not received) and attempts > 0:
        attempts -= 1
        time.sleep(1)

    assert ws.is_open()
    assert received

    # close()/stop() を呼ばずに参照を切る
    ws = None
    server = None
    client = None

    print("Success")
