import gc
import subprocess
import sys
import threading
import time
import weakref
from pathlib import Path

from libdatachannel import WebSocket, WebSocketConfiguration


# https://github.com/paullouisageneau/libdatachannel/blob/0e40aeb058b947014a918a448ce2d346e6ab14fe/test/websocket.cpp
# を Python に直したもの
def test_websocket(echo_websocket_server) -> None:
    my_message = "Hello world from libdatachannel"
    config = WebSocketConfiguration()
    config.disable_tls_verification = True
    ws = WebSocket(config)

    received = False
    opened = threading.Event()
    received_event = threading.Event()
    closed = threading.Event()

    def ws_on_open() -> None:
        assert ws is not None
        ws.send(my_message)
        opened.set()

    def ws_on_error(error: str) -> None:
        pass

    def ws_on_closed() -> None:
        closed.set()

    def ws_on_message(message: str | bytes) -> None:
        nonlocal received
        if isinstance(message, str):
            received = message == my_message
            if received:
                received_event.set()
            else:
                pass

    ws.on_open(ws_on_open)
    ws.on_error(ws_on_error)
    ws.on_closed(ws_on_closed)
    ws.on_message(ws_on_message)

    ws.open(echo_websocket_server)

    # callback から通知されるまで待つ (ポーリングしない)
    assert opened.wait(timeout=20), "WebSocket が open にならなかった"
    assert received_event.wait(timeout=20), "メッセージを受信しなかった"

    assert ws.is_open()
    assert received

    ws.close()
    assert closed.wait(timeout=20), "WebSocket が close にならなかった"

    # これが無いとリークする
    ws = None


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
    opened = threading.Event()
    closed = threading.Event()
    received_event = threading.Event()

    def ws_on_open() -> None:
        assert ws is not None
        # 先頭 3 バイトだけ送る (旧 (data, size) 版の size=3 と等価)
        ws.send(my_message[:3])
        opened.set()

    def ws_on_error(error: str) -> None:
        errors.append(error)

    def ws_on_closed() -> None:
        closed.set()

    def ws_on_message(message: str | bytes) -> None:
        if isinstance(message, bytes):
            received.extend(message)
            received_event.set()
        else:
            # エコーがバイナリで返らない場合はテストを失敗させる
            unexpected.append(message)

    ws.on_open(ws_on_open)
    ws.on_error(ws_on_error)
    ws.on_closed(ws_on_closed)
    ws.on_message(ws_on_message)

    ws.open(echo_websocket_server)

    # callback から通知されるまで待つ (ポーリングしない)
    assert opened.wait(timeout=20), "WebSocket が open にならなかった"
    assert received_event.wait(timeout=20), "メッセージを受信しなかった"

    assert not errors
    assert not unexpected
    assert received == b"012"

    ws.close()
    assert closed.wait(timeout=20), "WebSocket が close にならなかった"

    # これが無いとリークする
    ws = None


# 恒停を再現する検証は tests/hang_reproduction_websocket.py を子プロセスで実行する。
# 恒停した場合、 main thread が native の mutex 待ちになり pytest-timeout の SIGALRM は
# 発火しないため、 pytest プロセス内では実行せず subprocess.run の timeout で打ち切る。
#
# close() は対向の close handshake の完了を待つため、 push し続けるサーバー相手の実測では
# 1 回あたり 10 秒程度かかる。
_CLOSE_ITERATIONS = 1
_FORCE_CLOSE_ITERATIONS = 5
# 子プロセスの正当な待ち時間の最悪値 (接続待ち + callback 待ち) より十分大きい値にする。
_HANG_REPRODUCTION_TIMEOUT = 180


def _run_hang_reproduction(mode: str, iterations: int) -> subprocess.CompletedProcess[str]:
    """恒停し得る検証を子プロセスに分離して実行する"""
    script = Path(__file__).with_name("hang_reproduction_websocket.py")
    return subprocess.run(
        [sys.executable, str(script), mode, str(iterations)],
        capture_output=True,
        text=True,
        timeout=_HANG_REPRODUCTION_TIMEOUT,
        check=False,
    )


def test_close_does_not_hang_while_receiving() -> None:
    """受信 callback の実行中に close() を呼んでも hang せず Closed に到達することを検証する

    close() が GIL を保持したまま close 経路に入ると恒停するため、 恒停の窓を作る条件で
    GIL 解放下で実行されることを確かめる。
    """
    result = _run_hang_reproduction("close", _CLOSE_ITERATIONS)
    assert result.returncode == 0, (
        f"close で hang した可能性がある: returncode={result.returncode} "
        f"stderr={result.stderr[-2000:]}"
    )


def test_force_close_does_not_hang() -> None:
    """受信 callback の実行中に force_close() を呼んでも hang しないことを検証する

    同じ恒停の窓で、 forceClose() が GIL 解放下で実行され Closed に到達することを確かめる。
    """
    result = _run_hang_reproduction("force_close", _FORCE_CLOSE_ITERATIONS)
    assert result.returncode == 0, (
        f"force_close で hang した可能性がある: returncode={result.returncode} "
        f"stderr={result.stderr[-2000:]}"
    )


def test_del_releases_native() -> None:
    """callback 未登録の最小ケースで WebSocket が破棄されることを検証する

    Free Threading 環境では refcount=0 の即時 destruct 保証が弱いので、 gc.collect() を
    介して確実に破棄させる (PeerConnection の同名テストと同じ理由)。
    """
    ws = WebSocket()
    ref = weakref.ref(ws)
    ws = None
    gc.collect()
    assert ref() is None, "破棄が完了しなかった"


def test_close_is_idempotent(echo_websocket_server) -> None:
    """接続を開いた状態で close() を 2 回呼んでも 2 回目が早期 return で即時完了することを検証する

    未接続の WebSocket は初期状態が Closed で 1 回目も 2 回目も早期 return になるため、
    実際に接続して state が Closed 以外の状態から close() を呼ぶ。
    """
    ws = WebSocket()
    opened = threading.Event()
    ws.on_open(opened.set)
    ws.open(echo_websocket_server)

    # callback から通知されるまで待つ (ポーリングしない)
    assert opened.wait(timeout=20), "WebSocket が open にならなかった"
    assert ws.is_open()

    ws.close()
    assert ws.ready_state() is WebSocket.State.Closed
    start = time.monotonic()
    ws.close()
    elapsed = time.monotonic() - start
    # 2 回目は state==Closed 早期 return で即時完了する。 0.5 秒は CI ばらつきを
    # 許容しつつ 30 秒タイムアウトの regression を検出できる値。
    assert elapsed < 0.5
    assert ws.ready_state() is WebSocket.State.Closed
