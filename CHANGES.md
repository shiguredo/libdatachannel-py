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
- [FIX] PeerConnection を明示的に close() せずに破棄したときに Python プロセスが停止する問題を修正する
  - 従来は破棄時の C++ デストラクタが GIL 保持下で内部処理を実行するため、 内部処理が呼ぶコールバックが GIL 待ちで止まり Python プロセスが永続停止していた
  - `PeerConnection.__del__` で `close()` を自動的に呼ぶようにする
  - 明示 `close()` を GIL 解放下で実行し、 close 処理の完了 (Closed 状態) まで待機する
  - 待機が 30 秒で完了しなかった場合は `RuntimeWarning` を出す
  - なお、 コールバック内でブロッキング I/O を行うシナリオでは 30 秒タイムアウトに到達する場合があり、 完全な解消にはなっていない (根本解消は今後の課題)
  - @sile

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
