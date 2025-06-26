import asyncio
import threading
from queue import Queue

import pytest
from aiohttp import web


@pytest.fixture
def echo_websocket_server():
    """WebSocketエコーサーバーのフィクスチャ"""
    
    async def websocket_handler(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                await ws.send_str(msg.data)
            elif msg.type == web.WSMsgType.BINARY:
                await ws.send_bytes(msg.data)
        
        return ws
    
    app = web.Application()
    app.router.add_get("/", websocket_handler)
    
    # ポート番号を共有するためのキュー
    port_queue = Queue()
    runner = None
    loop = None
    
    def run_server():
        nonlocal runner, loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def start():
            nonlocal runner
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, '127.0.0.1', 0)
            await site.start()
            # 実際のポート番号を取得
            port = site._server.sockets[0].getsockname()[1]
            port_queue.put(port)
        
        loop.run_until_complete(start())
        loop.run_forever()
    
    # サーバーをバックグラウンドスレッドで起動
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    
    # ポート番号を取得
    port = port_queue.get(timeout=5)
    url = f"ws://127.0.0.1:{port}/"
    
    yield url
    
    # クリーンアップ
    if loop:
        loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5)