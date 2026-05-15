# 変更履歴

- CHANGE
  - 後方互換性のない変更
- UPDATE
  - 後方互換性がある変更
- ADD
  - 後方互換性がある追加
- FIX
  - バグ修正

## develop

- [FIX] PeerConnection / WebSocket / WebSocketServer を close()/stop() せずに destruct したときに hang する問題を修正する
  - PeerConnection は destructor 内 `mProcessor.join()` が GIL 保持下で blocking していた。 WebSocket / WebSocketServer も同じ系統の問題を持つ可能性があるため、 対称的に対処する
  - 各 close()/stop() の binding を GIL release で同期実行する形に変更する
  - Python 側で各クラスを wrapper class にラップし、 `__del__` で close()/stop() を呼ぶようにする
  - 公開クラス `libdatachannel.PeerConnection / WebSocket / WebSocketServer` は Python wrapper class に置き換わる (`isinstance` は維持されるが `type()` 厳密比較は変わる)
  - `WebSocketServer.on_client` callback に渡される `WebSocket` は wrapper ではなく native class インスタンスなので、 callback 内で保持する場合は明示的に `close()` を呼ぶことを推奨する
  - destruct hang のリグレッション検知用に `tests/test_peerconnection.py::test_destruct_without_explicit_close` と `tests/test_websocketserver.py::test_destruct_without_explicit_close` を追加する
  - @sile
- [UPDATE] cmake の最小バージョンを 4.3 にする
  - @voluntas
- [UPDATE] scikit-build-core の最小バージョンを 0.12.0 にする
  - @voluntas
- [UPDATE] nanobind の最小バージョンを 2.12.0 にする
  - @voluntas
- [ADD] Python 3.14t に対応する
  - Free Threading 対応
  - @voluntas
- [ADD] Python 3.12 に対応する
  - @voluntas

### misc

- [FIX] 依存ライブラリのビルドキャッシュのキーに Python バージョンを追加する
  - @voluntas
- [CHANGE] auditwheel の使用方法を uvx コマンドに変更する
  - @voluntas

## 2025.1.2

**リリース日**:: 2025-11-25

- [FIX] nanobind で DataChannelInit と LocalDescriptionInit がデフォルト引数としてモジュールに保持されリークする問題を修正する
  - @voluntas
- [FIX] MediaHandler チェーンのメモリーリークを修正は不要だったので revert する
  - @voluntas

## 2025.1.1

**リリース日**:: 2025-11-25

- [FIX] MediaHandler チェーンのメモリーリークを修正する
  - `track.close()` をオーバーライドして、 MediaHandler チェーンもクリアするようにする
  - @voluntas

## 2025.1.0

**リリース日**:: 2025-11-25

**祝いリリース**
