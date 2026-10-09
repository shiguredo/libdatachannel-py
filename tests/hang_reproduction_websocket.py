"""WebSocket の close 経路の恒停を再現する検証スクリプト

tests/test_websocket.py から子プロセスとして起動する。 pytest はこのファイルを
テストとして collect しない (ファイル名が test_ で始まらない)。

恒停は「内部 thread が受信 callback を実行して callback mutex を保持したまま GIL を
待っている間に、 GIL を保持した側が close 経路で同じ mutex を待つ」窓で発生する。
そのため 1 ms 間隔で push し続けるサーバーを立て、 受信 callback の実行中に
close() / force_close() を呼ぶ。 未修正の binding は GIL を保持したまま close 経路
(closeTransports()) に入るため恒停する。

恒停すると main thread が native の mutex 待ちになり pytest-timeout の SIGALRM は
発火しない。 そのためこの検証は pytest プロセス内では実行せず、 テスト側の
subprocess.run の timeout で打ち切る。

callback 内では print を使わない (callback 内 print の除去を進めているため使わない)。

最後は os._exit(0) で終了する。 C++ 側の public ~WebSocket() (rtc::WebSocket の
デストラクタ) は GIL を保持したまま走るため、 callback 実行中の窓では依然として
恒停し得る (根本対応は別 issue)。 ここでは close 経路の検証に絞る。
"""

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
# 接続待ちと callback 実行待ちの上限。 恒停した場合はテスト側の timeout で打ち切る。
WAIT_SECONDS = 5.0


def find_free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


PORT = find_free_port()


async def handler(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    try:
        while not ws.closed:
            await ws.send_str("x")
            await asyncio.sleep(0.001)
    except (ConnectionResetError, asyncio.CancelledError, RuntimeError):
        # 対向が close したときの送信失敗は無視する (push を続けることだけが目的)
        pass
    return ws


async def start_server() -> None:
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
# Event.wait() は GIL を解放するため callback 側の進行を妨げない。
callback_running = threading.Event()


def on_message(message: str | bytes) -> None:
    global received
    received += 1
    callback_running.set()


for i in range(ITERATIONS):
    ws = WebSocket()
    ws.on_message(on_message)
    callback_running.clear()
    ws.open(f"ws://127.0.0.1:{PORT}/")
    deadline = time.monotonic() + WAIT_SECONDS
    while not ws.is_open() and time.monotonic() < deadline:
        time.sleep(0.01)
    if not ws.is_open():
        print("接続に失敗しました", file=sys.stderr)
        sys.exit(1)
    # 受信 callback が実行中になるまで待ってから close 経路を呼ぶ
    if not callback_running.wait(timeout=WAIT_SECONDS):
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
        print(f"状態が Closed ではありません: {ws.ready_state()}", file=sys.stderr)
        sys.exit(1)

# C++ 側の ~WebSocket() の恒停 (別 issue) を避けるため、 明示的に即時終了する
os._exit(0)
