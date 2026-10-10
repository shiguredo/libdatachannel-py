"""IceUdpMuxListener の callback の例外でプロセスが落ちないかを確認するスクリプト

STUN Binding Request を mux へ送り、 例外を投げる callback を登録して、 プロセスが
落ちずに終了するかどうかを終了コードで確かめる。 std::terminate はプロセスごと落とす
ため、 pytest からは subprocess で起動する (tests/hang_reproduction_websocket.py と同じ方式)。
"""

import socket
import struct
import sys
import threading
import time

from libdatachannel import IceUdpMuxListener

# STUN の Binding Request とマジッククッキー
STUN_BINDING_REQUEST = 0x0001
STUN_MAGIC_COOKIE = 0x2112A442
# STUN の属性タイプ
STUN_ATTR_USERNAME = 0x0006
STUN_ATTR_MESSAGE_INTEGRITY = 0x0008
# MESSAGE-INTEGRITY は HMAC-SHA1 の 20 byte
MESSAGE_INTEGRITY_SIZE = 20
# callback が呼ばれるまでの待ち時間 (秒)
CALLBACK_TIMEOUT = 10


def build_stun_binding_request() -> bytes:
    """USERNAME と MESSAGE-INTEGRITY を付けた STUN Binding Request を組み立てる

    mux は Binding Request のうち USERNAME に ':' を含み、 20 byte の
    MESSAGE-INTEGRITY を持つものを未処理の STUN request として callback へ渡す。
    """
    username = b"abcd:efgh"
    attributes = struct.pack("!HH", STUN_ATTR_USERNAME, len(username)) + username
    # 属性は 4 byte 境界に揃える
    attributes += b"\x00" * (-len(username) % 4)
    integrity = b"\x00" * MESSAGE_INTEGRITY_SIZE
    attributes += struct.pack("!HH", STUN_ATTR_MESSAGE_INTEGRITY, len(integrity)) + integrity

    header = struct.pack("!HHI", STUN_BINDING_REQUEST, len(attributes), STUN_MAGIC_COOKIE)
    # transaction id は 12 byte
    return header + b"\x00" * 12 + attributes


def find_free_udp_port() -> int:
    """空いている UDP ポートを探す

    IceUdpMuxListener は port 0 を受け付けないため、 一度 bind して空きを確かめる。
    """
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def main() -> int:
    port = find_free_udp_port()
    listener = IceUdpMuxListener(port, "127.0.0.1")
    called = threading.Event()

    def on_unhandled_stun_request(request: object) -> None:
        called.set()
        # C のフレームを横断させないことを確かめるため、 ここで例外を投げる
        raise RuntimeError("callback の例外 (検証用)")

    listener.on_unhandled_stun_request(on_unhandled_stun_request)

    payload = build_stun_binding_request()
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.sendto(payload, ("127.0.0.1", listener.port()))

    # callback は C の thread から呼ばれるため、 呼ばれるまで待つ
    if not called.wait(timeout=CALLBACK_TIMEOUT):
        print("callback was not called", file=sys.stderr)
        listener.stop()
        return 1

    print("callback called", flush=True)
    # 例外を握り潰した後の状態を確かめるため、 少しだけ待つ
    time.sleep(0.2)
    listener.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
