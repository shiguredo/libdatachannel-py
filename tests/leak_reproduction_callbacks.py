"""callback のリークを確認するスクリプト

callback が wrapper 自身を捕捉する (閉包や束縛メソッド) と、 nanobind の
std::function への変換が callable を強参照するため、
nb_inst → C++ → std::function → callable → nb_inst の循環ができ、 Python の GC から
見えないところでインスタンスが残る。 close() を呼べば C++ 側の std::function が
解放されて循環が切れる。

nanobind のリーク警告はインタプリタ終了時に stderr へ出るため、 pytest からは
subprocess で起動する (tests/crash_reproduction_iceudpmuxlistener.py と同じ方式)。
"""

import gc
import sys

from libdatachannel import PeerConnection


class CallbackHolder:
    """wrapper を保持する callback を持つヘルパー

    束縛メソッドを callback に登録すると callback → self → wrapper の参照ができ、
    C++ 側の std::function が callable を強参照していることと合わせて循環が閉じる。
    """

    def __init__(self, peer_connection: PeerConnection) -> None:
        self._peer_connection = peer_connection
        self.last_state: object = None

    def on_state(self, state: object) -> None:
        """状態を受け取る callback (self 経由で wrapper を保持する)"""
        self.last_state = state


def main() -> int:
    pc = PeerConnection()
    holder = CallbackHolder(pc)

    # wrapper 自身を捕捉する callback を登録する (循環を作る)
    pc.on_state_change(holder.on_state)
    pc.on_ice_state_change(holder.on_state)

    # close() で C++ 側の std::function が解放される
    pc.close()
    del holder
    del pc
    gc.collect()

    print("closed and collected", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
