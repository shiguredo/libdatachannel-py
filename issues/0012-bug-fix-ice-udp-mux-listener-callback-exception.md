# IceUdpMuxListener の callback で例外が libjuice の C フレームを横断する

- Priority: Medium
- Created: 2026-08-30
- Completed: {YYYY-MM-DD}
- Branch: feature/fix-ice-udp-mux-listener-callback-exception
- Polished: {YYYY-MM-DD}

## 目的

IceUdpMuxListener の callback は libjuice のパケット受信スレッドから try/catch なしで直呼びされる。Python callback が例外を投げると `nb::python_error` が C のフレームを横断し、未定義動作 (典型的には std::terminate によるプロセス即死) になる。他の callback 経路と同様に例外を捕まえる。

## 優先度根拠

- Channel / PeerConnection / WebSocketServer の callback は libdatachannel 側の try/catch 内で呼ばれるが、この経路だけ捕まっていない
- 影響範囲は UDP MUX 利用者に限定されるが、落ち方がプロセス即死であり debug 困難
- 実測は未実施 (STUN request の注入が必要) である点は、本 issue がコード上の裏取りベースであることを意味する

## 現状

- `bind_iceudpmuxlistener` の on_unhandled_stun_request は libdatachannel の `IceUdpMuxListener::OnUnhandledStunRequest` を直接バインドする
- libdatachannel の `src/iceudpmuxlistener.cpp` は callback を libjuice の `conn_mux.c` から直呼びで呼ぶ
- Python callback の例外は nanobind が C++ 例外に変換して伝播するため、try/catch がないと C のフレームを横断する

## 設計方針

- binding 側で callback を try/catch する wrapper に置き換え、例外をログ (英語) に変換する
- 実測テストを追加する: localhost に STUN request を送信し、例外を投げる callback を登録して、プロセスが落ちないことを検証する

## 完了条件

- callback 内で例外を投げてもプロセスが落ちないこと
- 実測テストが追加されていること
- `uv sync && make test` で全テストが PASS すること
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること

## 参考

- 対象シンボル: `bind_iceudpmuxlistener` (src/bind_libdatachannel.cpp)
- libdatachannel v0.24.0: `src/iceudpmuxlistener.cpp`、`deps/libjuice/src/conn_mux.c`
- 関連 issue: [[0004-bug-fix-ice-udp-mux-listener-destructor-gil-release]] (同一クラスの destructor 問題。本 issue は callback 例外経路であり重複しない)
