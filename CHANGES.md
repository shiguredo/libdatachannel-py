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

- [CHANGE] `DataChannel.send()` / `Track.send()` / `WebSocket.send()` の `(data, size)` 版と、 `Track.send_frame()` の `(data, size, info)` 版を削除する
  - `size` に `data` の長さを超える値を渡すとヒープの範囲外を読み、 その内容が対向に送信されていた (SIGBUS でプロセスが落ちることもあった)
  - `size` は `len(data)` から導出できるため、 size を取らない版 (`send` は 1 引数、 `send_frame` は `data` と `info`) とスライス (`data[:size]`) で置き換えられる (`size` を渡して呼ぶと `TypeError` になる)
  - `Channel` 側の同じ版は binding の削除 (develop の `[FIX]` エントリ) で既に対応済み
  - @voluntas
- [UPDATE] cmake の最小バージョンを 4.3 にする
  - @voluntas
- [UPDATE] scikit-build-core の最小バージョンを 1.1.1 にする
  - @voluntas
- [UPDATE] nanobind の最小バージョンを 3.1.0 にする
  - nanobind 3 では `NB_TRAMPOLINE` の size 引数が不要になったため削除し、 型 caster の `flags` を `uint32_t` に広げて `noexcept` を付ける
  - `nb::gil_scoped_acquire::is_valid()` を使い、 interpreter 停止中は Python API を触らずに終了するようにする (Python 3.15 以降で必要)
  - @voluntas
- [ADD] Python 3.14t に対応する
  - Free Threading 対応
  - @voluntas
- [ADD] Python 3.12 に対応する
  - @voluntas
- [FIX] Description.media() / Description.application() / Media.rtp_map() / PeerConnection.config() が親の寿命に紐付かない参照を返す問題を修正する
  - `media()` / `application()` / `config()` の戻り値が親を生存させるようにする (親を先に破棄すると use-after-free で落ちていた)
  - `rtp_map()` は内部への参照ではなく値 (コピー) を返すようにする (`remove_rtp_map()` で無効になっていた)
  - @voluntas
- [FIX] MediaHandler の chain に cycle を作ると SEGV する問題を修正する
  - `add_to_chain` / `set_next` / `Track.chain_media_handler` で cycle を検出し、 連結する前に例外にする
  - cycle があると `MediaHandler::last()` が next() を無限に再帰してスタックオーバーフローで落ちていた
  - 検査の走査には上限 (1024 ノード) があり、 連結するチェーンがこの長さまでに終端へ到達しない場合は例外になる
  - @voluntas
- [FIX] WebSocket の close() / force_close() の GIL 保持による Python プロセスの停止を修正する
  - 従来は GIL を保持したまま close 経路に入り、 受信 callback を実行中の内部 thread とロック順逆転して Python プロセスが停止していた
  - `WebSocket.close()` / `WebSocket.force_close()` を GIL 解放下で実行し、 close() は close 処理の完了 (Closed 状態) まで待機するようにする
  - 待機が 30 秒で完了しなかった場合は `RuntimeWarning` を出す
  - `state` が `Closing` の場合は polling せず即 return する (対向の close handshake 完了は別 thread が行うため)
  - 対向の応答によっては close() が 10 秒程度ブロックする場合がある
  - なお、 明示 close() を呼ばずに破棄した場合の停止 (破棄時の C++ デストラクタが GIL 保持下で走るため) は解消していない (根本解消は今後の課題)
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
- [FIX] NalUnit と H265NalUnit にヘッダサイズ未満のバッファを渡すと SIGSEGV する問題を修正する
  - `NalUnit(0).forbidden_bit()` や `NalUnit(b"").forbidden_bit()` が 0 バイトのバッファで null ポインタを参照して落ち、 `H265NalUnit(1)` は 2 バイトのヘッダを範囲外で読み書きしていた (Release ビルドでは assert が消えるため libdatachannel 本体の防御が働かない)
  - コンストラクタ (size 版 / bytes 版) でヘッダサイズ以上のバッファが確保されるかを検証し、 範囲外と桁あふれは `ValueError` にする
  - @voluntas

### misc

- [FIX] CI の pytest リトライが job を救済できていなかったのを解消する
  - 1 回目の pytest ステップに `continue-on-error` を付け、 `steps.pytest.outcome` で失敗を検出して pytest を 1 回だけ再実行する
  - 型検査 (ty) はリトライの対象から外し、 失敗した場合は job を失敗させる
  - リトライが走ったことが分かるように `::warning::` の注記を出す
  - @voluntas
- [FIX] CI で pytest が実行されていなかったのを解消し、 ビルドした wheel を検証する
  - build_ubuntu / build_macos で wheel を fresh な環境に install してテストする
  - prek.toml に pytest のフックを追加し、 CI では wheel をビルドするジョブで実行する
  - 動かない build_debug.yml と、 参照されていない composite action を削除する
  - @voluntas
- [FIX] CI で prek.toml のフック (ruff / ty / tombi / clang-format と組み込みフック) が実行されていなかったのを解消する
  - pull_request と develop への push で prek.toml のフックを実行する
  - ty はビルドで生成されるスタブを必要とするため、 wheel をビルドするジョブで実行する
  - @voluntas
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
