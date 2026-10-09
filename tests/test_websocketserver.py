import gc
import socket
import sys
import threading
import time
import weakref

import pytest

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
            assert client is not None
            path = client.path()
            if path is not None:
                print(f"WebSocketServer: Requested path is {path}")

        def client_on_closed():
            print("WebSocketServer: Client connection closed")

        def client_on_message(message):
            nonlocal client
            assert client is not None
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
        assert ws is not None
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


def test_stop_releases_gil() -> None:
    """WebSocketServer.stop() が GIL を解放して実行されること

    受け入れ thread が Python callback の GIL を待っている間に stop() が GIL を
    保持したまま走ると、 Python プロセス全体が恒停し得る。 stop() の呼び出し中に
    GIL を待つ thread が進行するかで判定する。
    """
    # free-threading ビルドには GIL が無いため、 GIL 解放そのものを測れない
    # (sys._is_gil_enabled は 3.13 以降にしか無い)
    is_gil_enabled = getattr(sys, "_is_gil_enabled", lambda: True)
    if not is_gil_enabled():
        pytest.skip("GIL が無いビルド (free-threading) では GIL 解放を測れない")

    config = WebSocketServerConfiguration()
    config.port = 48090
    config.bind_address = "127.0.0.1"
    server = WebSocketServer(config)

    counter = 0
    stopping = threading.Event()

    def spin() -> None:
        nonlocal counter
        while not stopping.is_set():
            counter += 1

    original_interval = sys.getswitchinterval()
    thread = threading.Thread(target=spin, daemon=True)
    thread.start()
    try:
        # 待機 thread が実際に動き始めるまで待つ (起動前に測ると 0 のままになる)
        deadline = time.monotonic() + 5
        while counter == 0 and time.monotonic() < deadline:
            time.sleep(0)
        assert counter > 0, "GIL を待つ thread が動き始めなかった"

        # 定期切替を止め、 GIL を解放しない限り待機 thread が動けないようにする
        sys.setswitchinterval(1.0)

        # GIL を解放しない呼び出し (port()) は待機 thread に GIL を渡さないため
        # 即座に戻る
        baseline_start = time.monotonic()
        for _ in range(20):
            server.port()
        baseline_elapsed = time.monotonic() - baseline_start

        # stop() が GIL を解放すると、 待機 thread が switch interval (1 秒) の間 GIL を
        # 握るため、 呼び出しは GIL の再取得待ちで 1 秒近くかかる
        stop_start = time.monotonic()
        server.stop()
        stopped_elapsed = time.monotonic() - stop_start

        assert stopped_elapsed > 0.1, (
            "stop() が GIL を解放しなかった "
            f"(stopped_elapsed={stopped_elapsed:.3f}, baseline_elapsed={baseline_elapsed:.6f})"
        )
    finally:
        sys.setswitchinterval(original_interval)
        stopping.set()
        thread.join(timeout=5)
        # 失敗時もサーバーを確実に停止する (stop は冪等)
        server.stop()


def test_destruct_without_explicit_close() -> None:
    """明示 stop() を呼ばずに破棄しても恒停せず終了すること

    破棄経路では libdatachannel 本体の公開デストラクタが stop() を呼ぶ。 ここでは
    クライアントを接続しない状態で破棄し、 破棄が完了すること (Python オブジェクトが
    解放され weakref が死ぬこと) を確認する。

    クライアントが接続しておらず Python callback が GIL を待っていないため、 この
    テストは破棄経路の恒停を検出しない (恒停が起きればテスト自体が停止する)。 恒停の
    検出には GIL を待つ callback を動かす必要があり、 破棄経路の恒停の根本対応は
    別 issue で扱う。
    """
    config = WebSocketServerConfiguration()
    config.port = 48091
    config.bind_address = "127.0.0.1"
    server = WebSocketServer(config)
    ref = weakref.ref(server)

    assert ref() is not None

    # 明示 stop() を呼ばずに破棄する
    del server
    gc.collect()

    assert ref() is None, "WebSocketServer が破棄されなかった"


def test_del_calls_stop_on_python_subclass() -> None:
    """Python サブクラスでは __del__ から binding の stop が呼ばれること

    nanobind の tp_dealloc は C++ destructor を直接呼ぶため基底クラスのインスタンスでは
    __del__ は実行されないが、 Python サブクラスでは __del__ が実行される。 __del__ が
    正常に動くこと (GIL 解放下の stop を呼べること) を確認する。
    """
    del_called = []
    del_errors = []

    class Server(WebSocketServer):
        def __del__(self) -> None:
            del_called.append(True)
            # binding の __del__ (GIL 解放下の stop) が例外を投げても CPython は
            # unraisable として記録するだけでテストは PASS してしまうため、 ここで
            # 捕まえて検証する
            try:
                super().__del__()
            except BaseException as e:  # noqa: BLE001 (破棄経路の例外を検証する)
                del_errors.append(repr(e))
            # stop() が実際に呼ばれたことを、 停止後に接続できないことで確認する
            # (binding の __del__ は C++ の stop を直接呼ぶため、 Python 側の
            #  stop override では検出できない)
            try:
                with socket.create_connection(("127.0.0.1", self.port()), timeout=1):
                    del_errors.append("stop 後も接続できた")
            except OSError:
                pass

    config = WebSocketServerConfiguration()
    config.port = 48092
    config.bind_address = "127.0.0.1"
    server = Server(config)

    del server
    gc.collect()

    assert del_called == [True]
    assert del_errors == [], f"binding の __del__ が例外を投げた: {del_errors}"
