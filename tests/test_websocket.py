import gc
import subprocess
import sys
import time
import weakref

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


# 恒停 (GIL と libdatachannel の callback mutex のロック順逆転) の窓を作る検証スクリプト。
#
# 恒停は「内部 thread が受信 callback を実行して callback mutex を保持したまま GIL を
# 待っている間に、 GIL を保持した側が close 経路で同じ mutex を待つ」窓で発生する。
# そのため 1 ms 間隔で push し続けるサーバーを立て、 callback の実行中に close() /
# force_close() を呼ぶ。 未修正の binding は GIL を保持したまま close 経路
# (closeTransports()) に入るため恒停する。
#
# 恒停した場合、 main thread が native の mutex 待ちになるため pytest-timeout の SIGALRM は
# 発火しない。 そのためこの検証は pytest プロセス内では実行せず、 子プロセスに分離して
# 親側の subprocess.run の timeout で打ち切る。 callback 内では print を使わない
# (callback 内 print の除去を進めているため使わない)。
#
# 子プロセスは最後に os._exit(0) で終了する。 C++ 側の public ~WebSocket()
# (rtc::WebSocket のデストラクタ) は GIL を保持したまま走るため、 callback が実行中の
# 窓では依然として恒停し得る (根本対応は別 issue)。 ここでは close 経路の検証に絞る。
_HANG_REPRODUCTION_SCRIPT = """
import asyncio
import os
import socket
import sys
import threading
import time

from aiohttp import web

from libdatachannel import WebSocket

MODE = sys.argv[1]
ITERATIONS = int(sys.argv[2])


def find_free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


PORT = find_free_port()


async def handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    try:
        while not ws.closed:
            await ws.send_str("x")
            await asyncio.sleep(0.001)
    except Exception:
        pass
    return ws


async def start_server():
    app = web.Application()
    app.router.add_get("/", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", PORT)
    await site.start()


loop = asyncio.new_event_loop()
loop.run_until_complete(start_server())
threading.Thread(target=loop.run_forever, daemon=True).start()

received = 0
# callback が実行中であることを main thread に伝えるためのイベント。
# callback 先頭で set し、 main thread は wait() で同期してから close 経路を呼ぶ。
# Event.wait() は GIL を解放するため、 callback 側の進行を妨げない。
callback_running = threading.Event()


def on_message(message):
    global received
    received += 1
    callback_running.set()


for i in range(ITERATIONS):
    ws = WebSocket()
    ws.on_message(on_message)
    callback_running.clear()
    ws.open("ws://127.0.0.1:%d/" % PORT)
    deadline = time.monotonic() + 10.0
    while not ws.is_open() and time.monotonic() < deadline:
        time.sleep(0.01)
    if not ws.is_open():
        print("接続に失敗しました", file=sys.stderr)
        sys.exit(1)
    # 受信 callback が実行中になるまで待ってから close 経路を呼ぶ
    if not callback_running.wait(timeout=10.0):
        print("受信 callback が実行されませんでした", file=sys.stderr)
        sys.exit(1)
    if MODE == "force_close":
        # GIL 解放下で forceClose() が呼ばれることを検証する
        ws.force_close()
    else:
        # GIL 解放下で close() が呼ばれることを検証する
        ws.close()
    # どちらの経路も Closed に到達することを検証する
    if ws.ready_state() is not WebSocket.State.Closed:
        print("状態が Closed ではありません: %s" % ws.ready_state(), file=sys.stderr)
        sys.exit(1)

# C++ 側の ~WebSocket() の恒停 (別 issue) を避けるため、 明示的に即時終了する
os._exit(0)
"""

# close() は対向の close handshake を待つため 1 回あたり 10 秒程度かかる。
_CLOSE_ITERATIONS = 1
_FORCE_CLOSE_ITERATIONS = 5


def _run_hang_reproduction(mode: str, iterations: int) -> subprocess.CompletedProcess[str]:
    """恒停し得る検証を子プロセスに分離して実行する。

    子プロセスが恒停した場合は親側で subprocess.TimeoutExpired になりテストが失敗する
    ため、 CI の job timeout までは停止しない。
    """
    return subprocess.run(
        [sys.executable, "-c", _HANG_REPRODUCTION_SCRIPT, mode, str(iterations)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_close_does_not_hang_while_receiving() -> None:
    """受信 callback の実行中に close() を呼んでも hang せず Closed に到達することを検証する

    close() が GIL を保持したまま close 経路に入ると、 callback mutex を保持した内部 thread が
    GIL を待つため恒停する。 GIL 解放下で実行されることを、 恒停の窓を作る条件で確かめる。
    """
    result = _run_hang_reproduction("close", _CLOSE_ITERATIONS)
    assert result.returncode == 0, (
        f"close で hang した可能性がある: returncode={result.returncode} "
        f"stderr={result.stderr[-2000:]}"
    )


def test_force_close_does_not_hang() -> None:
    """受信 callback の実行中に force_close() を呼んでも hang しないことを検証する

    forceClose() も GIL を保持したまま実行すると closeTransports() で恒停し得るため、
    GIL 解放下で実行されることを同じ条件で確かめる。
    """
    result = _run_hang_reproduction("force_close", _FORCE_CLOSE_ITERATIONS)
    assert result.returncode == 0, (
        f"force_close で hang した可能性がある: returncode={result.returncode} "
        f"stderr={result.stderr[-2000:]}"
    )


def test_del_releases_native() -> None:
    """callback 未登録の最小ケースで __del__ 経由の close を検証する

    Free Threading 環境では refcount=0 の即時 destruct 保証が弱いので、 gc.collect() を
    介して確実に発火させる (0001 の PeerConnection の同名テストと同じ理由)。
    """
    ws = WebSocket()
    ref = weakref.ref(ws)
    ws = None
    gc.collect()
    assert ref() is None, "__del__ が発火しなかった"


def test_close_is_idempotent(echo_websocket_server) -> None:
    """接続を開いた状態で close() を 2 回呼んでも 2 回目が早期 return で即時完了することを検証する

    未接続の WebSocket は初期状態が Closed で 1 回目も 2 回目も早期 return になるため、
    実際に接続して state が Closed 以外の状態から close() を呼ぶ。
    """
    ws = WebSocket()
    ws.open(echo_websocket_server)

    attempts = 20
    while not ws.is_open() and attempts > 0:
        attempts -= 1
        time.sleep(1)
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
