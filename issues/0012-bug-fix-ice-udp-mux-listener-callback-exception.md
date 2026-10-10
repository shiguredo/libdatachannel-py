# IceUdpMuxListener の callback で例外が libjuice の C フレームを横断する

- Priority: Medium
- Created: 2026-08-30
- Completed: {YYYY-MM-DD}
- Branch: feature/fix-ice-udp-mux-listener-callback-exception
- Polished: 2026-10-10

## 目的

IceUdpMuxListener の callback は libjuice のパケット受信スレッドから try/catch なしで直呼びされる。Python callback が例外を投げると `nb::python_error` が C のフレームを横断し、未定義動作 (典型的には std::terminate によるプロセス即死) になる。他の callback 経路と同様に例外を捕まえる。

## 優先度根拠

- Channel / PeerConnection / WebSocketServer の callback は libdatachannel 側の try/catch 内で呼ばれるが、この経路だけ捕まっていない
- 影響範囲は UDP MUX 利用者に限定されるが、落ち方がプロセス即死であり debug 困難
- 実測は未実施 (STUN request の注入が必要) である点は、本 issue がコード上の裏取りベースであることを意味する

## 現状

- `bind_iceudpmuxlistener` の on_unhandled_stun_request は libdatachannel の `IceUdpMuxListener::OnUnhandledStunRequest` を直接バインドする
- libdatachannel の `src/impl/iceudpmuxlistener.cpp` は callback を libjuice の `conn_mux.c` から直呼びで呼ぶ
- Python callback の例外は nanobind が C++ 例外に変換して伝播するため、try/catch がないと C のフレームを横断する

## 設計方針

- binding 側で callback を try/catch する wrapper に置き換える。 `python_error` 以外の例外も拾えるよう `catch (...)` で受ける
- ログのレベルと文言は [[0015-bug-fix-callback-exception-handling]] が定める統一の方針に合わせる (0015 に「IceUdpMuxListener 経路は 0012 で対応する」と切り分け済み)。 binding には現在ログ機構が無く、 既存の握り潰しは `PyErr_WarnEx(PyExc_RuntimeWarning, ...)` のみである点を踏襲する
- callback は mux の registry mutex を保持した状態で呼ばれるため、 callback から `stop()` などを呼ぶとデッドロックし得る。 例外を投げるだけのテストにする
- 実測テストを追加する。 localhost の mux port に STUN Binding Request (USERNAME に `:` を含む値と 20 byte の MESSAGE-INTEGRITY を付ける) を送ると `OnUnhandledStunRequest` が呼ばれる。 例外を投げる callback を登録して、 プロセスが落ちないことを検証する
  - クラッシュは pytest プロセスごと落とすため、 `tests/crash_reproduction_iceudpmuxlistener.py` を用意し、 `tests/test_iceudpmuxlistener.py` から `subprocess.run` + timeout で起動して終了コードを検証する (`tests/hang_reproduction_websocket.py` と同じ方式。 `pytest.mark.timeout` はネイティブ側の停止では発火しない)

## 完了条件

- callback 内で例外を投げてもプロセスが落ちないこと (終了コード 0 で終わること)
- 上記の実測テストが追加されていること
- `make develop` のあと `uv run --no-sync python -m pytest tests/ -q --deselect tests/test_peerconnection.py::test_destruct_without_explicit_close`、 `prek run --all-files pytest`、 `prek run --all-files ty` が PASS すること (`uv sync && make test` は install を壊し、 恒停するテストも実行するため使わない)
- `CHANGES.md` の `## develop` に `[FIX]` として記録すること
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること

## 参考

- 対象シンボル: `bind_iceudpmuxlistener` (src/bind_libdatachannel.cpp)
- libdatachannel v0.24.0: `src/impl/iceudpmuxlistener.cpp`、`deps/libjuice/src/conn_mux.c`
- 関連 issue: [[0004-bug-fix-ice-udp-mux-listener-destructor-gil-release]] (同一クラスの destructor 問題。本 issue は callback 例外経路であり重複しない)
