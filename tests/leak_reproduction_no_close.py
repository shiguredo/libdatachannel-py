"""close() を呼ばない場合に callback の循環が回収されることを確認するスクリプト

callback が wrapper 自身を捕捉する (閉包 / 束縛メソッド) と、 callback を C++ 側で
強参照している場合は nb_inst → C++ → std::function → callable → nb_inst の循環が
でき、 C++ 側が Python の GC の対象外であるためにインスタンスが解放されない
(issue 0053)。 callback を Python 側 (インスタンスの __dict__) で強参照し、 C++ 側では
弱参照で持つようにしたため、 close() を呼ばなくても循環が GC から見えるようになり、
gc.collect() でインスタンスを回収できる。

nanobind のリーク警告はインタプリタ終了時に stderr へ出るため、 pytest からは
subprocess で起動する (tests/leak_reproduction_callbacks.py と同じ方式)。
"""

import gc
import sys
import weakref

from libdatachannel import PeerConnection


class CallbackHolder:
    """wrapper を保持する callback を持つヘルパー

    束縛メソッドを callback に登録すると callback → self → wrapper の参照ができ、
    callback を C++ 側で強参照していると循環が閉じてインスタンスが解放されない。
    """

    def __init__(self, peer_connection: PeerConnection) -> None:
        self._peer_connection = peer_connection
        self.last_state: object = None

    def on_state(self, state: object) -> None:
        """状態を受け取る callback (self 経由で wrapper を保持する)"""
        self.last_state = state


def register_callbacks_and_drop() -> weakref.ref[PeerConnection]:
    """wrapper を捕捉する callback を登録し、 close() を呼ばずに局所変数を手放す

    関数を抜けると局所変数からの参照が消え、 callback と wrapper の相互参照だけが
    残る。 close() を呼ばないため C++ 側の std::function が callback を強参照して
    いると循環が切れず、 インスタンスは解放されない。
    """
    pc = PeerConnection()
    holder = CallbackHolder(pc)

    # wrapper 自身を捕捉する callback を登録する (循環を作る)
    pc.on_state_change(holder.on_state)
    pc.on_ice_state_change(holder.on_state)
    # inline の lambda も閉包で holder 経由の循環を作る
    pc.on_local_description(lambda description: holder.on_state(description))

    return weakref.ref(pc)


def main() -> int:
    ref = register_callbacks_and_drop()

    gc.collect()

    if ref() is not None:
        print(
            "gc.collect() しても PeerConnection が解放されていない",
            file=sys.stderr,
            flush=True,
        )
        return 1

    print("close() を呼ばずに解放された", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
