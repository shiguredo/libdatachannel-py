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

- [FIX] PeerConnection を close() せずに destruct すると GIL を保持したまま hang する問題を修正する
  - libdatachannel の `~PeerConnection()` 内 `mProcessor.join()` が非同期処理 (SCTP shutdown 等) の完了を同期待ちするが、 binding 側で GIL を保持したまま呼ばれるため別スレッドが動けず終了できなかった
  - 対策 1: `PeerConnection.close()` の binding を lambda にして `nb::call_guard<nb::gil_scoped_release>()` で GIL を release しつつ state が Closed になるまで polling する形に変更する
  - 対策 2: Python 側で `PeerConnection` を wrapper class でラップし、 `__del__` で `close()` を呼んでから C++ destructor に渡すようにする
  - これにより `tests/test_peerconnection.py::test_leak` の skip も解除する
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
