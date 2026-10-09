# nanobind の破棄経路では __del__ が呼ばれないため破棄時の保護が機能しない

- Priority: High
- Created: 2026-10-09
- Completed: {YYYY-MM-DD}
- Branch: feature/fix-nanobind-del-not-called
- Polished: {YYYY-MM-DD}

## 目的

nanobind で作成した binding に `.def("__del__", ...)` を追加しても、 Python の object 破棄時には実行されない。 このため「破棄時に GIL 解放下で `close()` と callback の解除を行う」という保護が実際には機能しておらず、 破棄経路の恒停が残っている。 影響範囲を確定し、 破棄時の保護を実現する方法を決める。

## 優先度根拠

- [[0001-bug-fix-peer-connection-destructor-gil-release]] (対応済み) が `PeerConnection.__del__` を追加しているが、 同じ理由で破棄時には呼ばれない。 既存の破棄テスト (`tests/test_peerconnection.py::test_destruct_without_explicit_close`) は恒停するため `--deselect` で除外されており、 機能していないことを検出できていない
- [[0003-bug-fix-websocketserver-destructor-gil-release]] / [[0004-bug-fix-ice-udp-mux-listener-destructor-gil-release]] も同じ方式を前提にすると、 同じ問題を抱える
- 利用者が `ws = None` で破棄する経路は Python では一般的であり、 恒停すると Python プロセス全体が停止する

## 現状

- nanobind の `inst_dealloc` (nanobind の `src/nb_type.cpp`) は `t->destruct(p)` で C++ destructor を直接呼び、 CPython の finalizer (`PyObject_CallFinalizerFromDealloc`) を呼ばない。 nanobind のソースに `__del__` の出現は 0 件
- したがって `.def("__del__", ...)` は通常のメソッドとして生えるだけで、 `ws = None` / `del ws` では実行されない。 `nb::type_slots` で `Py_tp_finalize` を登録しても、 nanobind の dealloc からは呼ばれない
- 実測 ([[0002-bug-fix-websocket-destructor-gil-release]] の作業中): `del ws` の間、 別 thread の Python カウンタが完全に凍結する (GIL が解放されていない = `__del__` が実行されていない)。 `ws.__del__()` を手で呼ぶと `state` は `Closed` になり、 続く `del ws` も完走する (guard 自体は有効)
- 実測: 1 ms 間隔で push し続けるサーバー相手では、 修正後も `del ws` で恒停する。 スタックは main thread が `~WebSocket()` → `closeTransports()` → `Transport::onRecv()` → `std::recursive_mutex::lock()`、 poll thread が `Channel::flushPendingMessages()` → `PyGILState_Ensure` → `take_gil`
- `tests/test_websocket.py::test_del_releases_native` は weakref が消えることしか見ていないため、 この問題に対しては false positive になる (破棄自体は nanobind が必ず行う)

## 設計方針 (案)

- 方針 A: Python wrapper class を復活させ、 wrapper の `__del__` (Python 定義のクラスなので確実に呼ばれる) から GIL 解放付きの `close()` と callback 解除を呼ぶ。 [[0001-bug-fix-peer-connection-destructor-gil-release]] で一度撤回された方式だが、 nanobind の制約により再検討が必要
- 方針 B: 破棄経路の恒停そのものを解消する ([[0005-bug-fix-destructor-callback-deadlock]] と統合する)
- どちらを採るかは [[0005-bug-fix-destructor-callback-deadlock]] とまとめて判断する。 方針 A を採る場合、 `PeerConnection` / `WebSocketServer` / `IceUdpMuxListener` の `__del__` binding も同時に見直す

## 完了条件

- 破棄経路 (`ws = None` / `pc = None`) で保護処理が実際に実行されることを実測で確認する (テストで検出できる形にする)
- `PeerConnection` / `WebSocket` の `__del__` binding の扱い (削除するか維持するか) を確定し、 issue と CHANGES に反映する
- 既存の破棄テストが `--deselect` なしで PASS するか、 恒停が残る場合はその理由と回避策が issue に明記されている
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること

## スコープ外 (関連する未解決問題)

- 破棄経路の恒停そのものの根本対応は [[0005-bug-fix-destructor-callback-deadlock]] に集約する
- 送信系 API の GIL 保持は [[0032-bug-fix-send-gil-deadlock]] で対応済み

## 参考

- nanobind の `inst_dealloc`: `_deps` ではなくビルド時に取得される nanobind パッケージの `src/nb_type.cpp`
- 関連 issue: [[0002-bug-fix-websocket-destructor-gil-release]] / [[0005-bug-fix-destructor-callback-deadlock]] / [[0001-bug-fix-peer-connection-destructor-gil-release]] / [[0003-bug-fix-websocketserver-destructor-gil-release]] / [[0004-bug-fix-ice-udp-mux-listener-destructor-gil-release]]
