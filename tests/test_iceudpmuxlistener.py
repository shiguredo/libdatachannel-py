"""IceUdpMuxListener のテスト

明示 stop() を呼ばずに破棄した場合の恒停を防ぐ仕組み (GIL 解放下の stop) を検証する。
"""

import gc
import sys
import threading
import time
import weakref

import pytest

from libdatachannel import IceUdpMuxListener


def _is_gil_enabled() -> bool:
    """GIL が有効か (sys._is_gil_enabled は 3.13 以降にしか無い)"""
    return getattr(sys, "_is_gil_enabled", lambda: True)()


def test_stop_releases_gil() -> None:
    """IceUdpMuxListener.stop() が GIL を解放して実行されること

    内部 thread が Python callback の GIL を待っている間に stop() が GIL を保持したまま
    走ると、 Python プロセス全体が恒停し得る。 stop() の呼び出し中に GIL を待つ thread が
    進行するかで判定する。
    """
    # free-threading ビルドには GIL が無いため、 GIL 解放そのものを測れない
    if not _is_gil_enabled():
        pytest.skip("GIL が無いビルド (free-threading) では GIL 解放を測れない")

    listener = IceUdpMuxListener(48095, "127.0.0.1")

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
        # 待機 thread に新しい switch interval で GIL を待たせ直す。 ここで一度 GIL を
        # 手放して保留中の受け渡しを解消する。 これをしないと、 待機 thread は変更前の
        # 短い interval (既定 5 ms) で待ち続けているため、 計測中に周期的な受け渡しが
        # 起きて、 GIL を解放しない呼び出しでも進行が観測されてしまう
        time.sleep(0)

        # GIL を解放しない呼び出し (port()) は待機 thread に GIL を渡さない。
        # この baseline は失敗時の診断用で、 判定には使わない (待機 thread の待ち直しが
        # 効かない環境では baseline 中にも受け渡しが起き得るため)
        baseline_start = time.monotonic()
        for _ in range(20):
            listener.port()
        baseline_elapsed = time.monotonic() - baseline_start

        # GIL を解放しない限り、 待機 thread は switch interval (1 秒) のあいだ GIL を
        # 得られない。 1 秒より十分短い 50 ms のあいだ呼び続け、 その間に待機 thread が
        # 進行すれば解放されていると判定する (stop は冪等)
        stopped = 0
        stopped_start = counter
        deadline = time.monotonic() + 0.05
        while time.monotonic() < deadline:
            listener.stop()
            stopped = counter - stopped_start
            if stopped:
                break

        assert stopped > 0, (
            f"stop() が GIL を解放しなかった "
            f"(stopped={stopped}, baseline_elapsed={baseline_elapsed:.6f})"
        )
    finally:
        sys.setswitchinterval(original_interval)
        stopping.set()
        thread.join(timeout=5)
        # 失敗時も listener を確実に停止する (stop は冪等)
        listener.stop()


def test_destruct_without_explicit_close() -> None:
    """明示 stop() を呼ばずに破棄しても恒停せず終了すること

    破棄経路では libdatachannel 本体の公開デストラクタが stop() を呼ぶ。 ここでは
    明示 stop() を呼ばずに破棄し、 破棄が完了すること (Python オブジェクトが解放され
    weakref が死ぬこと) を確認する。

    Python callback が GIL を待っていないため、 このテストは破棄経路の恒停を検出しない
    (恒停が起きればテスト自体が停止する)。 恒停の検出には GIL を待つ callback を動かす
    必要があり、 破棄経路の恒停の根本対応は別 issue で扱う。
    """
    listener = IceUdpMuxListener(48096, "127.0.0.1")
    ref = weakref.ref(listener)

    assert ref() is not None

    # 明示 stop() を呼ばずに破棄する
    del listener
    gc.collect()

    assert ref() is None, "IceUdpMuxListener が破棄されなかった"


def test_del_calls_stop_on_python_subclass() -> None:
    """Python サブクラスでは __del__ から binding の stop が呼ばれること

    nanobind の tp_dealloc は C++ destructor を直接呼ぶため基底クラスのインスタンスでは
    __del__ は実行されないが、 Python サブクラスでは __del__ が実行される。 破棄時に
    binding の __del__ (GIL 解放下の stop) が例外なく呼べることを確認する。 binding から
    __del__ を削除すると super().__del__() が AttributeError になるため、 この検証で
    呼ばれたことが分かる。
    """
    del_called = []
    del_errors = []

    class Listener(IceUdpMuxListener):
        def __del__(self) -> None:
            del_called.append(True)
            # binding の __del__ (GIL 解放下の stop) が例外を投げても CPython は
            # unraisable として記録するだけでテストは PASS してしまうため、 ここで
            # 捕まえて検証する。 binding から __del__ を削除すると AttributeError に
            # なるため、 この検証で binding の __del__ が呼ばれたことが分かる。
            #
            # stop() が実際に走ったことは Python からは観測できない。 UDP ポートを
            # bind し直せるかで判定しようとすると、 libjuice が mux socket を registry
            # に保持し接続中の agent が無くなるまで cleanup しないため、 stop() 直後に
            # 同じポートを bind できるとは限らず (CI の Linux leg で失敗した)、 判定に
            # 使えない
            try:
                super().__del__()
            except BaseException as e:  # noqa: BLE001 (破棄経路の例外を検証する)
                del_errors.append(repr(e))

    listener = Listener(48097, "127.0.0.1")

    del listener
    gc.collect()

    assert del_called == [True]
    assert del_errors == [], f"binding の __del__ が異常終了した: {del_errors}"
