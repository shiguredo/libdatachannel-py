"""callback のライフタイム (issue 0053) の回帰テスト

callback を Python 側 (インスタンスの __dict__) で強参照し、 C++ 側では弱参照で持つ
ようにした。 これにより

- callback が wrapper 自身を捕捉していても、 close() を呼ばずに破棄して
  gc.collect() すればインスタンスが回収される (子プロセスで実行して確認する)
- inline の lambda や束縛メソッドを登録しても、 従来どおり callback が呼ばれる

の 2 点を確認する。
"""

import subprocess
import sys
import threading
from pathlib import Path

from libdatachannel import PeerConnection, WebSocket, WebSocketConfiguration

# 子プロセスの正当な実行時間より十分大きい値
_LEAK_REPRODUCTION_TIMEOUT = 60

# callback から通知されるまで待つ上限
_CALLBACK_TIMEOUT = 20


def _run_leak_reproduction() -> subprocess.CompletedProcess[str]:
    """close() を呼ばない callback リークの再現スクリプトを子プロセスで実行する"""
    script = Path(__file__).with_name("leak_reproduction_no_close.py")
    return subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=_LEAK_REPRODUCTION_TIMEOUT,
        check=False,
    )


def test_callback_closure_is_released_without_close() -> None:
    """callback が wrapper を捕捉していても close() なしで回収されること

    nanobind の std::function caster は callable を強参照するため、 wrapper を捕捉する
    callback を登録したまま close() を呼ばないと、 Python の GC から見えない循環ができて
    インスタンスが解放されず、 終了時に "nanobind: leaked" が stderr へ出る。
    """
    result = _run_leak_reproduction()

    assert result.returncode == 0, (
        f"リークの再現スクリプトが失敗した: returncode={result.returncode} "
        f"stderr={result.stderr[-2000:]}"
    )
    assert "nanobind: leaked" not in result.stderr, (
        f"callback を登録したインスタンスが解放されていない: stderr={result.stderr[-2000:]}"
    )


class _CallbackRecorder:
    """束縛メソッドを callback として登録するヘルパー"""

    def __init__(self, peer_connection: PeerConnection) -> None:
        self._peer_connection = peer_connection
        self.states: list[object] = []
        self.ice_states: list[object] = []

    def on_state(self, state: object) -> None:
        """状態を受け取る callback"""
        self.states.append(state)

    def on_ice_state(self, state: object) -> None:
        """ICE の状態を受け取る callback"""
        self.ice_states.append(state)


def test_inline_lambda_callback_is_called() -> None:
    """inline の lambda を登録しても従来どおり呼ばれること

    lambda への参照はインスタンスの __dict__ (Python 側) にだけ存在する。 Python 側で
    強参照していなければ C++ 側の弱参照は登録直後に切れ、 callback は呼ばれない。
    """
    pc = PeerConnection()
    states: list[object] = []

    pc.on_state_change(lambda state: states.append(state))

    # close() は Closed への状態遷移を同期的に通知する
    pc.close()

    assert states == [PeerConnection.State.Closed], (
        f"inline の lambda が呼ばれていない: states={states}"
    )
    assert pc.state() is PeerConnection.State.Closed


def test_bound_method_callback_is_called() -> None:
    """束縛メソッドを登録しても従来どおり呼ばれること

    束縛メソッドは callback → self → wrapper の参照を持つため、 Python 側で強参照して
    いなければ登録した wrapper ごと循環して解放されなくなる。 呼ばれることも合わせて
    確認する。
    """
    pc = PeerConnection()
    recorder = _CallbackRecorder(pc)

    pc.on_state_change(recorder.on_state)
    pc.on_ice_state_change(recorder.on_ice_state)

    pc.close()

    assert recorder.states == [PeerConnection.State.Closed], (
        f"状態の callback が呼ばれていない: states={recorder.states}"
    )
    assert recorder.ice_states == [PeerConnection.IceState.Closed], (
        f"ICE 状態の callback が呼ばれていない: ice_states={recorder.ice_states}"
    )


class _MessageReceiver:
    """受信したメッセージを保持するヘルパー"""

    def __init__(self) -> None:
        self.binary_messages: list[bytes] = []
        self.text_messages: list[str] = []
        self.received = threading.Event()

    def on_binary(self, message: bytes) -> None:
        """binary のメッセージを受け取る callback"""
        self.binary_messages.append(message)
        self.received.set()

    def on_text(self, message: str) -> None:
        """string のメッセージを受け取る callback"""
        self.text_messages.append(message)
        self.received.set()


def test_channel_binding_callback_is_called(echo_websocket_server: str) -> None:
    """基底クラス (Channel) の binding に登録した callback が呼ばれること

    on_open / on_message は Channel の binding で、 WebSocket は Channel を 2 番目の
    基底に持つ。 callback を保持する Python オブジェクトを引く経路が壊れていないことを、
    実際に接続して確認する。 on_message は 2 つの callback を取る overload を使う。
    """
    config = WebSocketConfiguration()
    config.disable_tls_verification = True
    ws = WebSocket(config)

    opened = threading.Event()
    receiver = _MessageReceiver()

    # inline の lambda と束縛メソッドの両方を Channel の binding へ登録する
    ws.on_open(lambda: opened.set())
    ws.on_message(receiver.on_binary, receiver.on_text)

    ws.open(echo_websocket_server)
    assert opened.wait(timeout=_CALLBACK_TIMEOUT), "WebSocket が open にならなかった"

    message = "callback lifetime"
    ws.send(message)
    assert receiver.received.wait(timeout=_CALLBACK_TIMEOUT), "文字列を受信しなかった"
    assert receiver.text_messages == [message], (
        f"受信した文字列が異なる: messages={receiver.text_messages}"
    )

    receiver.received.clear()
    ws.send(b"\x01\x02\x03")
    assert receiver.received.wait(timeout=_CALLBACK_TIMEOUT), "バイナリを受信しなかった"
    assert receiver.binary_messages == [b"\x01\x02\x03"], (
        f"受信したバイナリが異なる: messages={receiver.binary_messages}"
    )

    ws.close()
