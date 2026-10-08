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
- [UPDATE] scikit-build-core の最小バージョンを 1.0.3 にする
  - @voluntas
- [UPDATE] nanobind の最小バージョンを 2.13.0 にする
  - @voluntas
- [ADD] Python 3.14t に対応する
  - Free Threading 対応
  - @voluntas
- [ADD] Python 3.12 に対応する
  - @voluntas
- [FIX] PeerConnection を明示的に close() せずに破棄したときに Python プロセスが停止する問題を修正する
  - 従来は破棄時の C++ デストラクタが GIL 保持下で内部処理を実行するため、 内部処理が呼ぶコールバックが GIL 待ちで止まり Python プロセスが永続停止していた
  - `PeerConnection.__del__` で `close()` を自動的に呼び、 close() 自身も GIL 解放下で close 処理の完了 (Closed 状態) まで待機するようにする
  - 待機が 30 秒で完了しなかった場合は `RuntimeWarning` を出す
  - なお、 コールバック内でブロッキング I/O を行うシナリオでは 30 秒タイムアウトに到達する場合があり、 完全な解消にはなっていない (根本解消は今後の課題)
  - @sile
- [FIX] 送信系 API が GIL を保持したまま送信経路に入り、 受信経路のコールバックとデッドロックする問題を修正する
  - 映像トラックとデータチャネルを同時に使い、 受信側のパケットロスが多い条件下で Python プロセスが恒久停止していた
  - `DataChannel.send()` / `Track.send()` / `Track.send_frame()` / `WebSocket.send()` を GIL 解放下で実行する
  - GIL による直列化が無くなるため、 複数 thread から同一 Track へ送信する場合は呼び出し側で直列化する
  - @voluntas
- [FIX] DataChannel / Track / WebSocket の `buffered_amount()` を Python から呼ぶと SIGSEGV する問題を修正する
  - `Channel` が 2 番目の基底であるため、 `Channel` 側の binding 経由では基底オフセットが加算されず、 virtual 呼び出しが誤った vtable スロットを読んでいた
  - `buffered_amount` を派生クラス側に binding し、 派生クラスのインスタンスから呼ぶ経路を修正する
  - @voluntas
- [FIX] `Channel` の binding 経由で virtual メソッドを呼ぶと落ちる、 または誤った関数が実行される問題を修正する
  - `Channel` は派生クラスの 2 番目の基底であるため、 `Channel` 側の binding 経由では基底オフセットが加算されず、 virtual 呼び出しが誤った vtable スロットを読んでいた
  - `Channel` の virtual メソッドの binding を削除する。 対象は close / send (2 オーバーロード) / is_open / is_closed / max_message_size / buffered_amount
  - 派生クラス側の binding は変更しない
  - 派生クラスのインスタンスからは従来どおり呼べる。 影響は未バインドで呼んでいたコードと、 型スタブから `Channel` のメソッドが消えることによる型検査 (`Channel` 型で注釈した変数からの呼び出し) に限られる
  - @voluntas

### misc

- [FIX] make lint / make typecheck が失敗したままになっていたのを解消し、 ruff の規約セットを明示して固定する
  - ruff / ty は prek.toml の rev でバージョンを管理し、 `[dependency-groups]` から削除する
  - @voluntas
- [FIX] Ubuntu 22.04 向け wheel ビルドで auditwheel 6.8.1 以降が要求する patchelf をインストールするようにする
  - @voluntas
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
