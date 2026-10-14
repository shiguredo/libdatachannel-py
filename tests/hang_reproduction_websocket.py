"""WebSocket の close 経路の恒停を再現する検証スクリプト

tests/test_websocket.py から子プロセスとして起動する。 pytest はこのファイルを
テストとして collect しない (ファイル名が test_ で始まらない)。

恒停は「内部 thread が受信 callback を実行して callback mutex を保持したまま GIL を
待っている間に、 GIL を保持した側が close 経路で同じ mutex を待つ」窓で発生する。
そのため 1 ms 間隔で push し続けるサーバーを立て、 受信 callback の実行中に
close() / force_close() を呼ぶ。 未修正の binding は GIL を保持したまま close 経路
(closeTransports()) に入るため恒停する。

接続確立は負荷の高い CI ランナーでは数秒かかることがあり、 一度の接続失敗で
検証全体を落とすと断続的に失敗する。 そのため接続は複数回試行し、 断続的な
セットアップの失敗を吸収する。 恒停の検出は親プロセスの subprocess timeout が
担うため、 ここでの再試行が恒停を隠すことはない (再試行中に恒停すれば timeout
になる)。

恒停すると main thread が native の mutex 待ちになり pytest-timeout の SIGALRM は
発火しない。 そのためこの検証は pytest プロセス内では実行せず、 テスト側の
subprocess.run の timeout で打ち切る。

callback 内では print を使わない (callback 内 print の除去を進めているため使わない)。

最後は os._exit(0) で終了する。 C++ 側の public ~WebSocket() (rtc::WebSocket の
デストラクタ) は GIL を保持したまま走るため、 callback 実行中の窓では依然として
恒停し得る (根本対応は別途行う)。 ここでは close 経路の検証に絞る。
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
# 接続確立の試行回数と 1 回あたりの接続待ちの上限。 負荷の高い CI ランナーでは
# open() から is_open() への反映が数秒遅れることがあるため、 接続失敗は
# 断続的なセットアップの失敗として複数回の試行で吸収する。
CONNECT_ATTEMPTS = 3
CONNECT_WAIT_SECONDS = 10.0
# 受信 callback の実行待ちの上限。 恒停した場合はテスト側の timeout で打ち切る。
CALLBACK_WAIT_SECONDS = 5.0


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
    except (ConnectionResetError, RuntimeError):
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

# callback が実行中であることを main thread に伝えるためのイベント。
# callback 先頭で set し、 main thread は wait() で同期してから close 経路を呼ぶ。
# Event.wait() は GIL を解放するため callback 側の進行を妨げない。
callback_running = threading.Event()

# 接続失敗の原因を切り分けるために on_error を記録する。 callback 内では print を
# 使わない方針のため、 記録だけして main thread 側でまとめて出力する。
connect_errors: list[str] = []


def on_message(message: str | bytes) -> None:
    # 恒停の窓を確実に作るため、 callback 内で GIL を解放する。
    # callback が即座に return する場合、 内部 thread が callback mutex を保持したまま
    # GIL を待つ状態がほぼ発生せず、 修正前でも恒停しない (回帰を検出できない)。
    callback_running.set()
    time.sleep(0.001)


def open_websocket() -> WebSocket | None:
    """WebSocket を open し、 is_open() になるまで待つ

    断続的な接続失敗を吸収するため CONNECT_ATTEMPTS 回まで試行する。 すべての
    試行で開けなかった場合は None を返す。
    """
    for _ in range(CONNECT_ATTEMPTS):
        ws = WebSocket()
        ws.on_message(on_message)
        ws.on_error(connect_errors.append)
        callback_running.clear()
        ws.open(f"ws://127.0.0.1:{PORT}/")
        deadline = time.monotonic() + CONNECT_WAIT_SECONDS
        while not ws.is_open() and time.monotonic() < deadline:
            time.sleep(0.01)
        if ws.is_open():
            return ws
        # 開けなかったソケットは残さず破棄する。 force_close() は同期で Closed に
        # 到達するため、 未接続ソケットの後始末に close() の polling は使わない。
        # このソケットでは callback が実行されないため恒停の窓は発生しない。
        ws.force_close()
    return None


for i in range(ITERATIONS):
    connect_errors.clear()
    ws = open_websocket()
    if ws is None:
        print(f"接続に失敗しました: errors={connect_errors}", file=sys.stderr)
        sys.exit(1)
    # 受信 callback が実行中になるまで待ってから close 経路を呼ぶ
    if not callback_running.wait(timeout=CALLBACK_WAIT_SECONDS):
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
