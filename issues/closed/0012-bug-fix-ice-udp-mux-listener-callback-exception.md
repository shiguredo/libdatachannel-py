# IceUdpMuxListener の callback で例外が libjuice の C フレームを横断する

- Priority: Medium
- Created: 2026-08-30
- Completed: 2026-10-10
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

## 解決方法

- `src/bind_libdatachannel.cpp`
  - `on_unhandled_stun_request` の binding を、 callback を try/catch で包む wrapper に置き換えた。 callback は libjuice の C callback (`conn_mux.c`) から `src/impl/iceudpmuxlistener.cpp` を経て直接呼ばれるため、 Python の例外が C のフレームを横断すると `std::terminate` になっていた (実測: exit 134)
  - 例外は `catch (...)` で受け止め、 `PyErr_WarnEx(PyExc_RuntimeWarning, ...)` で記録する (binding に既存の握り潰しと同じ方式)。 filterwarnings で warning が例外に昇格した場合も C のフレームへ例外を出さないよう `PyErr_Clear()` する
  - callback は mux の registry mutex を保持した状態で呼ばれるため、 callback から `stop()` を呼ぶとデッドロックし得る。 テストは例外を投げるだけにしている
- `tests/crash_reproduction_iceudpmuxlistener.py` (新規)
  - STUN Binding Request (USERNAME に `:` を含む値と 20 byte の MESSAGE-INTEGRITY) を mux の port へ送り、 例外を投げる callback が呼ばれたうえでプロセスが正常終了することを確かめるスクリプト
- `tests/test_iceudpmuxlistener.py`
  - 上記スクリプトを `subprocess.run` + timeout で起動し、 終了コード 0 と callback が呼ばれたことを検証するテストを追加した (`tests/hang_reproduction_websocket.py` と同じ方式。 `std::terminate` は pytest プロセスごと落とすため in-process では観測できない)
- `CHANGES.md`
  - `## develop` に `[FIX]` として記録した
- 検証
  - `tests/test_iceudpmuxlistener.py` 4 passed、 全体 152 passed / 12 skipped / 1 deselected
  - `prek run --all-files pytest` と `prek run --all-files ty` が PASS
  - 修正前は同じ手順で exit 134 (`libc++abi: terminating due to uncaught exception of type nanobind::python_error`) になることをレビューで実測済み
  - ログレベルと文言の統一 ([[0015-bug-fix-callback-exception-handling]]) は本 issue の範囲外で、 0015 側の作業として残る

## 参考

- 対象シンボル: `bind_iceudpmuxlistener` (src/bind_libdatachannel.cpp)
- libdatachannel v0.24.0: `src/impl/iceudpmuxlistener.cpp`、`deps/libjuice/src/conn_mux.c`
- 関連 issue: [[0004-bug-fix-ice-udp-mux-listener-destructor-gil-release]] (同一クラスの destructor 問題。本 issue は callback 例外経路であり重複しない)
