# PeerConnection を明示的に close() せずに destruct した場合の hang を修正する

- Priority: High
- Created: 2026-05-18
- Model: Opus 4.7
- Branch: feature/fix-peer-connection-destructor-gil-release

## 目的

`PeerConnection` を明示的に `close()` せずに destruct すると、 libdatachannel 本体の destruct 経路が Python オブジェクト破棄経路 (= GIL 保持中) で blocking 処理を走らせ、 Python プロセス全体が長時間 hang し得る。 本 issue ではこれを症状緩和する。 根本対応は [[0005-bug-fix-destructor-callback-deadlock]] に集約する。

## 優先度根拠

- `PeerConnection` は本ライブラリの中心 API であり、 暗黙の destruct 経路を踏む可能性がほぼ全シナリオで存在する。
- `tests/test_peerconnection.py:203-304` の `test_leak` は本問題のため `@pytest.mark.skip(reason="なぜかブロックして止まったままになるのでいったんスキップ")` されており、 リグレッション検知が機能していない。
- 同系統の後続 issue [[0002-bug-fix-websocket-destructor-gil-release]] (High) / [[0003-bug-fix-websocketserver-destructor-gil-release]] (Medium) / [[0004-bug-fix-ice-udp-mux-listener-destructor-gil-release]] (Medium) が、 本 issue の成果物 (共通ヘルパー / wrapper パターン / `pyproject.toml` 設定 / `[FIX]` 扱い判断) を踏襲する。 PR 戦略は「実装手順」 で詳述する。

## 現状

`src/bind_libdatachannel.cpp:1320` の `PeerConnection` bindings は `.def("close", &PeerConnection::close)` で直接バインドしており、 GIL を保持したまま libdatachannel の `close()` が実行される。 `src/libdatachannel/__init__.py` には Python wrapper class が存在せず、 明示 `close()` 忘れに対するセーフティネットも無い。

利用者が `del pc` あるいはスコープ抜けで暗黙の destruct を起こすと、 以下のチェーンで hang し得る。

1. public `~PeerConnection()` (`_deps/libdatachannel/v0.24.0/source/src/peerconnection.cpp:40-46`) が `impl()->remoteClose()` を呼ぶ。
2. impl `remoteClose()` (`_deps/.../src/impl/peerconnection.cpp:97-106`) は `state != Closed` の条件下で `closeDataChannels` / `closeTracks` を `mProcessor` に enqueue し、 `closeTransports()` で同期的に State::Closed まで進める。
3. impl `~PeerConnection()` (`_deps/.../src/impl/peerconnection.cpp:82-85`) は `mProcessor.join()` を Python オブジェクト destruct 経路 (= GIL 保持中) で実行する。
4. `mProcessor` の残タスクが Python callback (`closeTracks` 経由で `Track::close()` (`_deps/.../src/impl/track.cpp:68-77`) → 基底 `Channel::triggerClosed()` (`_deps/.../src/impl/channel.cpp:24-30`) → Python `on_closed` 等) を呼んで GIL 待ちに入ると、 destruct を呼んだ thread の GIL を握ったまま `mProcessor.join()` が return せず Python プロセス全体が永続 hang する。

## 設計方針

### 1. C++ binding 側 (`src/bind_libdatachannel.cpp`)

#### 1.1 症状緩和の原理

利用者が明示 `close()` を呼ばなくても wrapper の `__del__` が `close()` を GIL release 下で呼ぶことで、 destruct 到達時には既に `state == Closed` になっている (= 上記「現状」 step 2 の `state != Closed` 分岐をスキップする) ため、 destructor 経路で `mProcessor.enqueue(closeDataChannels, closeTracks)` が走らず、 `mProcessor.join()` が即時 return する確率が大幅に上がる。 加えて、 polling 中は GIL release されるため、 `mProcessor` 上の残タスクが Python callback を呼ぶ場合でも callback thread が GIL を取って進める。 この 2 点で多くの実利用シナリオで hang を回避する。

#### 1.2 共通 polling ヘルパー `wait_for_closed` (本 issue で先行導入)

匿名 namespace `namespace { ... }` (現状 `src/bind_libdatachannel.cpp:61` から始まる) 内、 `bind_configuration` (現状 `src/bind_libdatachannel.cpp:65`) の **直前** に以下を追加する。

```cpp
// SCTP / DTLS / TCP shutdown は典型的に数百 ms 〜 数秒で完了する。 これに対し
// 十分短く busy loop にもならない値として 10 ms を採用する。 後続の WebSocket
// でも同じ定数を共有する。
constexpr auto kPollInterval = std::chrono::milliseconds(10);
// libdatachannel の ICE / DTLS タイムアウト最大値を踏まえ、 ネットワーク
// 遅延・リトライを見込んでも余裕がある上限として 30 秒。 これを超えても
// Closed に到達しないケースは異常状態と判断し polling を諦め、 destructor
// 側の mProcessor.join() に委ねる。
constexpr auto kCloseTimeout = std::chrono::seconds(30);

// 呼び出し側 binding は `nb::call_guard<nb::gil_scoped_release>()` で GIL を
// release した状態で本関数を呼び、 polling ループ中も GIL を保持しない。
// `get_state(self)` は atomic load 等の軽量・ GIL 不要な操作であること
// (PeerConnection::state() / WebSocket::readyState() 等)。
//
// 注: 1 回目の `close()` が timeout した直後に `__del__` 経由で再度呼ばれる
// 直列ケースでは、 最悪 `kCloseTimeout × 2 = 60` 秒待ち得る。 libdatachannel
// 側の処理停滞時のみのレアケースで、 wrapper 側に重複抑制フラグを持たせる
// コストに見合わないため許容する。
template <typename Self, typename GetState, typename State>
void wait_for_closed(Self& self,
                     GetState get_state,
                     State closed,
                     const char* warning_msg) {
  const auto deadline = std::chrono::steady_clock::now() + kCloseTimeout;
  while (get_state(self) != closed) {
    if (std::chrono::steady_clock::now() >= deadline) {
      nb::gil_scoped_acquire gil;
      // filterwarnings=error 等で warning が例外昇格された場合は pending
      // exception を放置せず Python 例外として伝播させる。
      if (PyErr_WarnEx(PyExc_RuntimeWarning, warning_msg, 1) < 0) {
        throw nb::python_error();
      }
      return;  // polling は諦め、 残処理は destructor の mProcessor.join() に委ねる。
    }
    std::this_thread::sleep_for(kPollInterval);
  }
}
```

`State` を独立した template 引数にする理由は、 [[0002]] が `WebSocket::State::Closed` (= `PeerConnection::State` と別 enum 型) を渡せるようにするため。 `decltype(closed)` で推論することも可能だが、 enum 値の型を独立 parameter で受ける形にすると意図が明確。

#### 1.3 `close_peer_connection` の追加と `.def("close", ...)` の差し替え

同じ匿名 namespace 内、 `bind_peerconnection` (現状 `src/bind_libdatachannel.cpp:1270` 付近) の **直前** に以下を追加する。

```cpp
void close_peer_connection(PeerConnection& self) {
  // libdatachannel 側 close() は `closing.exchange(true)` (impl/peerconnection.cpp:88)
  // で冪等性が確保されているが、 polling 30 秒経由を避けるための早期 return。
  if (self.state() == PeerConnection::State::Closed) {
    return;
  }
  self.close();
  wait_for_closed(
      self,
      [](PeerConnection& s) { return s.state(); },
      PeerConnection::State::Closed,
      "PeerConnection.close(): state did not reach Closed within timeout; "
      "the remaining cleanup is delegated to the C++ destructor and may block.");
}
```

bindings の `.def("close", &PeerConnection::close)` を以下に差し替える。

```cpp
.def("close", &close_peer_connection, nb::call_guard<nb::gil_scoped_release>())
```

補足:

- `nb::call_guard<nb::gil_scoped_release>()` は `close` bindings に限定する。 `state()` / `config()` 等の頻繁に呼ぶ短時間 API は atomic load / 短い mutex 区間で長時間ブロックしないため GIL release を入れない。 `setLocalDescription` 等の長時間 API への GIL release は本 issue ではスコープ外。
- `State::Closed` 以外 (例: `Failed` / `Disconnected` / `Connecting`) で `close()` を呼んだ場合、 libdatachannel の `closing.exchange(true)` 経路で `mSctpTransport` の有無を判定し、 有: SCTP `stop()` → 非同期に state==Closed に遷移、 無: `remoteClose()` 同期実行で State::Closed まで進む。 polling は両経路を統一的に扱える。 異常状態で Closed に到達しないケースは `kCloseTimeout` で抜ける。
- 本 issue が `nb::call_guard<nb::gil_scoped_release>()` を libdatachannel-py で **初導入** する (現状の `bind_libdatachannel.cpp` に他用例なし)。 `<nanobind/nanobind.h>` は既に include 済みのため追加 include 不要。

#### 1.4 必要な include

`src/bind_libdatachannel.cpp` の include 群末尾 (`#include <rtc/rtc.hpp>` の直後) に `// 標準ライブラリ` コメント付きブロックを新設し `<chrono>` と `<thread>` を追加する。

### 2. Python wrapper 側 (`src/libdatachannel/__init__.py`)

`from .libdatachannel_ext import *` で持ち込まれた native `PeerConnection` を、 同名の wrapper class で後勝ち上書きする。

```python
from .libdatachannel_ext import *  # noqa: F401,F403
from .libdatachannel_ext import PeerConnection as _PeerConnection


class PeerConnection(_PeerConnection):  # type: ignore[misc]
    """PeerConnection の Python wrapper。

    明示的に ``close()`` を呼ぶことを推奨する。 ``__del__`` 内で ``close()``
    を呼ぶセーフティネットを備えるが、 ``__del__`` は GC タイミングに依存
    するため close 完了時刻は予測しにくく、 interpreter shutdown 時の挙動
    保証もできない。 例外を観測したい場合は明示 ``close()`` を呼ぶこと。
    """

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
```

補足:

- import 順序は「`*` import → `_PeerConnection` 退避 → wrapper class 定義」 で、 Python 名前解決の後勝ちで wrapper が公開名 `PeerConnection` を上書きする。 `_PeerConnection` 自体は module 属性として残るが、 `_` prefix で private を示す慣例に従う。
- `# type: ignore[misc]` は nanobind の native 型を Python 側で subclass する際の型チェッカ警告抑止用 (native 側に `nb::is_final()` 未指定で subclass 可能。 C++ `class PeerConnection final` は Python subclass には影響しない)。
- `except Exception: pass` の対象範囲は全 `Exception` (timeout 由来の `RuntimeWarning` の例外昇格、 `close()` 内 C++ 例外、 interpreter shutdown 中の属性消失による `TypeError` 等)。 silent failure になる trade-off を受け入れ、 利用者には docstring で明示 `close()` を強く推奨する。 `BaseException` ではなく `Exception` を握ることで `KeyboardInterrupt` / `SystemExit` は通す。

### 3. 後方互換と pyi スタブ

公開クラス `libdatachannel.PeerConnection` が wrapper class に置き換わる。 利用者影響:

- `isinstance(pc, PeerConnection)` は subclass 関係で従来通り動作。
- `type(pc) is PeerConnection` の厳密比較は結果が変わる (実利用での該当は想定しにくい)。
- `__module__` が `libdatachannel.libdatachannel_ext` → `libdatachannel` に変わる (`__qualname__` は両方とも `PeerConnection` で不変)。 ログ・traceback の表示が変わるが、 実利用への影響は小さい。
- 戻り値型 (`pc.create_data_channel()` 等) は wrapper を返さないので影響なし。

`_build/__init__.pyi` は `nanobind_add_stub` で自動生成され (`CMakeLists.txt:225-244`)、 `Makefile:8-9` で `src/libdatachannel/__init__.pyi` にコピーされる。 wrapper class は pyi に出現しないが、 wrapper は native と同じ public API を継承するため利用者コードの型 hint 精度には影響しない。 `__del__` 等 wrapper 内 method は型 checker から不可視。 本 issue では pyi の手動補追はスコープ外。

CHANGES.md は `[FIX]` で記載する。 `type()` 厳密比較・ `__module__` 表示の変化があるため厳密には後方互換に影響するが、 実利用での破壊リスクが低いと判断し `[CHANGE]` ではなく `[FIX]` で扱う。 同判断を [[0002]] / [[0003]] / [[0004]] にも適用する。

### 4. テスト (`tests/test_peerconnection.py`)

新規・改修 test は以下 3 つ。

#### 4.1 `test_destruct_without_explicit_close` (既存 `test_leak` を改修)

既存 `test_leak` (line 203-304) の `@pytest.mark.skip(reason="...")` を解除し、 `test_destruct_without_explicit_close` にリネームする。 改修内容:

- **`test_leak` 関数内の全 `print(...)` 呼び出し** (`print(..., file=sys.stderr)` を含む) **をコメントアウト**する。 理由コメントは 1 か所に集約 (各 callback ごとに繰り返さない)。

  ```python
  def pc1_on_state_change(state):
      # pytest stdout capture と組み合わせると [[0005]] の callback I/O block 経路を踏み
      # テストが hang するため print をコメントアウトする。 根本対応は 0005 を参照。
      # print("State 1: " + str(state))
      pass
  ```

  - `print` 以外の実処理 (`pc2.set_remote_description(...)` 等) は残す。 callback 本体が空になる場合は `pass` を入れる。
  - `import sys` は `test_track` (`tests/test_peerconnection.py:123`) で引き続き使われるため残す。

- 関数末尾を以下に変更する。 `recwarn` fixture (pytest 内蔵、 `import pytest` 不要) で `RuntimeWarning` 0 件を assert する。

  ```python
  def test_destruct_without_explicit_close(recwarn):
      # ... 既存の test_leak セットアップとアサーションがここに続く ...
      pc1 = None
      pc2 = None
      gc.collect()
      runtime_warnings = [w for w in recwarn.list if issubclass(w.category, RuntimeWarning)]
      assert not runtime_warnings, (
          f"close() の polling timeout 警告が {len(runtime_warnings)} 件発生: "
          f"{[str(w.message) for w in runtime_warnings]}"
      )
  ```

  - callback closure (`pc1` の callback が `pc2` を closure 参照、 逆も同様) で循環参照があり、 `pc1 = None; pc2 = None` だけでは reference counting が 0 にならないため `gc.collect()` を明示呼びする。
  - callback 内 `print` をコメントアウト済みのため、 `gc.collect()` で wrapper `__del__` が発火しても [[0005]] の callback I/O block 経路は踏まない想定。
  - `recwarn.list` には `__del__` 経由で発火した `RuntimeWarning` も記録される (`__del__` の `except Exception: pass` は warning 自体の発火 (`PyErr_WarnEx`) は阻害せず、 warning が例外昇格された場合の例外伝播を握り潰すだけ)。 これにより「30 秒 timeout 警告が出ても残り 30 秒で `mProcessor.join()` が完了し PASS」 という偽合格を防ぐ。
  - 想定外の挙動 (nanobind の callback wrapper が GC traversal に参加しない等で `gc.collect()` でも wrapper `__del__` が発火しない場合) では、 destruct が発生せず hang もしないため test は PASS する。 ただしこの場合 wrapper の動作確認にはならないため、 wrapper の `__del__` 経由 close 動作は 4.2 で別途確実に検証する。
  - `import gc` を test ファイル先頭に追加する (既存 import は `sys` / `time` / `pytest` の順なので、 `import sys` の前に置く)。

#### 4.2 `test_wrapper_del_releases_native` (新規)

callback を一切登録しない最小ケースで、 wrapper の `__del__` が確実に発火し native instance が解放されることを weakref で間接検証する。

```python
import weakref

def test_wrapper_del_releases_native():
    pc = PeerConnection()
    ref = weakref.ref(pc)
    pc = None
    # native PeerConnection には nb::is_weak_referenceable() 指定が無いため weakref 不可だが、
    # Python 側で class PeerConnection(_PeerConnection) と subclass しているため、 派生 class
    # には Python の type 機構が自動で __weakref__ slot を付与する。 したがって wrapper
    # instance に対する weakref は動作する。 wrapper の refcount が 0 になり __del__ →
    # close() → native 解放と進むと weakref が dead になる。
    assert ref() is None
```

callback closure が無いため refcount=0 経路で wrapper `__del__` が即発火し、 wrapper の `close()` 呼出し → native 解放 → weakref が dead になることを検証する。 これにより wrapper パターンの動作 (= 本 issue の中核実装) がリグレッション検知できる。

#### 4.3 `test_close_is_idempotent` (新規)

`close()` を 2 回呼んでも 2 回目が早期 return で即時完了することを検証する。

```python
def test_close_is_idempotent():
    pc = PeerConnection()
    pc.close()
    assert pc.state() is PeerConnection.State.Closed
    start = time.monotonic()
    pc.close()
    elapsed = time.monotonic() - start
    assert elapsed < 1.0
    assert pc.state() is PeerConnection.State.Closed
```

- `import time` は test ファイル先頭 (`tests/test_peerconnection.py:2`) で既に import 済み。
- 1 回目の `close()` は SCTP 未生成のため `remoteClose()` 同期実行で State::Closed まで進み、 polling は while 条件評価のみで即抜ける。 2 回目は state == Closed 早期 return で即時 return する。 polling ロジック自体は本 test 内では本格的に実行されない (polling 経路は実 E2E test (`test_track` 等) で間接的にカバー)。
- 1 秒閾値は CI ばらつきを許容する余裕値。 早期 return が壊れて 30 秒 timeout を踏むケースは確実に検出できる。

#### 4.4 import と pyproject.toml

skip 解除に伴い `@pytest.mark.skip` がなくなる。 `tests/test_peerconnection.py` 内で `pytest` は他で使われていないため `import pytest` を削除する。 代わりに `import gc` を追加する (`recwarn` は fixture として引数注入されるため `import pytest` 不要)。

`pyproject.toml` の既存 `[tool.pytest.ini_options]` セクション (現状 `pyproject.toml:86-87`) に `timeout = 60` を **既存セクション内に追記** する (セクション再宣言ではない)。

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
timeout = 60
```

`pytest-timeout` は既に `pyproject.toml` の依存に入っている。 `timeout_method` はデフォルト (signal) のまま。 既存 `test_track` (約 22 秒の sleep を含む) も 60 秒以内で完了する想定。 後続 [[0002]] / [[0003]] / [[0004]] のテストも本設定を共有する (= 後続 issue では `pyproject.toml` を再変更しない)。

### 5. CHANGES.md

`## develop` セクションに以下の `[FIX]` エントリを追加する。 既存 `## develop` の種別並びは [UPDATE] → [ADD] で CLAUDE.md 種別順序「CHANGE → ADD → UPDATE → FIX」 と一部不整合だが、 本 issue では既存違反 (line 14 の `[UPDATE}` 閉じ括弧 typo を含む) には触らない。 本 `[FIX]` は既存 `[ADD]` の後・ `### misc` の前に挿入する。

```
- [FIX] PeerConnection を明示的に close() せずに destruct した場合に発生する GIL 保持 hang を修正する
  - 公開クラス `PeerConnection` が Python wrapper に置き換わる (`isinstance` は維持、 `type()` 厳密比較は変わる)
  - `__del__` 経由で `close()` が自動的に呼ばれる
  - 明示 `close()` の挙動が変わる (GIL release で実行、 state==Closed まで polling)
  - timeout 30 秒で `RuntimeWarning` を出す
  - @<github-username>
```

担当者行は補足項目の後、 エントリの最後に書く。 commit 時に実装担当者が `<github-username>` を自分の GitHub アカウント名に置換する (本 issue ではプレースホルダ)。 `pyproject.toml` への `timeout = 60` 追加は本 `[FIX]` の test 追加に伴う付随変更とみなし、 `### misc` には別途記載しない。

## 完了条件

- `uv sync && make test` で全 test が PASS する。 特に:
  - `tests/test_peerconnection.py::test_destruct_without_explicit_close` が skip されず、 `pytest-timeout=60` の下で PASS し、 `RuntimeWarning` も発生しない。
  - `tests/test_peerconnection.py::test_wrapper_del_releases_native` が PASS し、 weakref が dead になる。
  - `tests/test_peerconnection.py::test_close_is_idempotent` が PASS し、 2 回目 `close()` が 1 秒以内に return する。
- `pyproject.toml` の `[tool.pytest.ini_options]` に `timeout = 60` が追加されている。
- `CHANGES.md` の `## develop` に `[FIX]` エントリが追加されている。
- `/review-diff-code` の致命的 / 重要指摘が 0 件。

## 実装手順 (実装担当者向け)

- ブランチ命名: `feature/fix-peer-connection-destructor-gil-release` (`feature/fix-` prefix、 ブランチ名に issue 番号を含めない)。
- PR 戦略: **本 issue を develop に先行 merge してから後続 [[0002]] / [[0003]] / [[0004]] を別 PR で着手する**。 本 issue が共通ヘルパー (`wait_for_closed`) / wrapper パターン / `pytest-timeout=60` 設定 / 後方互換性判断 (`[FIX]` 扱い) を導入し、 後続はこれらに依存する。 4 issue を 1 PR にまとめるのは履歴と review の見通しを下げるため避ける。
- 参考ブランチ `feature/fix-destructor-gil-release` は試行錯誤の履歴を含むため cherry-pick せず、 本 issue の設計方針通りに新規ブランチで実装する。
- 後続 issue で本 issue を踏襲する事項:

  | 踏襲事項 | [[0002]] | [[0003]] | [[0004]] |
  | --- | --- | --- | --- |
  | `wait_for_closed` の再利用 | あり | なし (state API 無) | なし (state API 無) |
  | wrapper class + `__del__` パターン | あり | あり | あり |
  | `pytest-timeout=60` の活用 (`pyproject.toml` は本 issue 済み) | あり | あり | あり |
  | `recwarn` による `RuntimeWarning` 0 件検証 | あり (polling あり) | なし (polling 無) | なし (polling 無) |
  | `[FIX]` 扱い + 後方互換影響の判断 | あり | あり | あり |

  これら踏襲事項について、 後続 issue 側にも「[[0001]] の規定に従う」 旨を明記して整合性を取ること。

## スコープ外 (関連する未解決問題)

本 issue は destruct 到達前に明示 `close()` で State::Closed まで進めて destructor 経路の負担を減らすアプローチであり、 完全には hang を防げない。 30 秒 timeout 後に `close_peer_connection` が return しても、 続く `~PeerConnection()` (impl) の `mProcessor.join()` は依然 GIL 保持下で走る。 callback I/O block 条件下では引き続き hang し得るため、 根本対応は [[0005-bug-fix-destructor-callback-deadlock]] に集約する。

`DataChannel.close()` / `Track.close()` / `Channel.close()` の binding にも理論上同様の GIL 保持下 close 問題が存在し得るが、 通常 `PeerConnection.close()` 経由で間接的に閉じられるため本 issue ではスコープ外とする。

## 参考

- 関連 issue: [[0002-bug-fix-websocket-destructor-gil-release]] / [[0003-bug-fix-websocketserver-destructor-gil-release]] / [[0004-bug-fix-ice-udp-mux-listener-destructor-gil-release]] / [[0005-bug-fix-destructor-callback-deadlock]]
- 「現状」 で未引用の libdatachannel v0.24.0 関連コード位置:
  - `_deps/libdatachannel/v0.24.0/source/src/peerconnection.cpp:48` (public `close()` で `impl()->close()` を呼ぶ)
  - `_deps/libdatachannel/v0.24.0/source/src/impl/peerconnection.cpp:87-95` (impl `close()` の `closing.exchange(true)` と SCTP stop / remoteClose 分岐)
  - `_deps/libdatachannel/v0.24.0/source/src/impl/peerconnection.cpp:336-353` (SCTP state-change callback で remoteClose を非同期 enqueue する switch ブロック)
  - `_deps/libdatachannel/v0.24.0/source/src/impl/peerconnection.cpp:1306-1329` (`changeState()` で State::Closed 時の同期 callback 実行)
  - `_deps/libdatachannel/v0.24.0/source/src/impl/peerconnection.cpp:377-389` (`closeTransports()` 内の `resetCallbacks()`)
  - `_deps/libdatachannel/v0.24.0/source/include/rtc/utils.hpp:79-82` (`synchronized_callback::operator()` の `recursive_mutex` 保持実行)
- libdatachannel-py 関連コード位置:
  - `src/bind_libdatachannel.cpp:61` (匿名 namespace の開始位置), `:65` (`bind_configuration` 開始), `:1270` (`bind_peerconnection` 開始), `:1320` (`.def("close", ...)`)
  - `src/libdatachannel/__init__.py` (wrapper 追加先)
  - `tests/test_peerconnection.py:1-15` (import), `:203` (`test_leak` の skip マーカ)
  - `pyproject.toml:86-87` (`[tool.pytest.ini_options]` セクション)
  - `CHANGES.md` `## develop` セクション
