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

- [CHANGE] `Media.as_audio()` / `Media.as_video()` が参照を返すようにする
  - 従来は値コピーを返しており、 戻り値への codec 追加が元の `Media` に反映されなかった
  - 動的型が一致する場合は同じオブジェクトを返し、 一致しない場合は `TypeError` になる (従来は未定義動作)
  - `Description` は media を `Description.Media` として保持するため (`add_media()` / `add_audio()` / SDP の parse はすべて `Media` にスライスされる)、 `Description` から取得した media では `TypeError` になる。 codec は `Description.Audio` / `Description.Video` に追加してから `add_media()` する (add_media 後の media へ後から codec を足すことはできない。 詳細は `[CHANGE] Description.media() / Description.application() が値 (コピー) を返すようにする` のエントリを参照)
  - @voluntas
- [CHANGE] `Description.media()` / `Description.application()` が値 (コピー) を返すようにする
  - `clear_media()` は内部の media と application を、 `add_media(Application)` / `add_application()` は内部の application を解放するため、 参照を返していると取得済みの戻り値が無効になり、 触ると use-after-free で落ちていた (実測: exit 139)
  - 戻り値を書き換えても `Description` に反映されなくなる。 media へ codec などを足す場合は、 codec を足した media を組み立ててから `add_media()` する (`add_rtp_map()` などは `Description.Media` のメソッドで、 `add_media()` のあとに取得したコピーへ足しても `Description` には反映されない)
  - @voluntas
- [CHANGE] `Media.rtp_map()` が値 (コピー) を返すようにする
  - 内部の `RtpMap` への参照は `remove_rtp_map()` / `remove_format()` で無効になっていた
  - 戻り値を書き換えても `Media` に反映されなくなる
  - 既存の payload type を変更する場合は `remove_rtp_map()` の後に `add_rtp_map()` を呼ぶ (`add_rtp_map()` は同じ payload type が既にあると上書きしない)
  - @voluntas
- [CHANGE] `DataChannel.send()` / `Track.send()` / `WebSocket.send()` の `(data, size)` 版と、 `Track.send_frame()` の `(data, size, info)` 版を削除する
  - `size` に `data` の長さを超える値を渡すとヒープの範囲外を読み、 その内容が対向に送信されていた (SIGBUS でプロセスが落ちることもあった)
  - `size` は `len(data)` から導出できるため、 size を取らない版 (`send` は 1 引数、 `send_frame` は `data` と `info`) とスライス (`data[:size]`) で置き換えられる (`size` を渡して呼ぶと `TypeError` になる)
  - `Channel` 側の同じ版は binding の削除 (develop の `[FIX]` エントリ) で既に対応済み
  - @voluntas
- [UPDATE] examples/whip.py と examples/whep.py が Trickle ICE の PATCH で local candidate を送るようにする
  - 201 Created を受信するまで candidate を保持し、 受信後に 1 つの HTTP PATCH (`application/trickle-ice-sdpfrag`) でまとめて送る (RFC 9725 Section 4.3.2 / draft-ietf-wish-whep-03 Section 4.4.2)
  - ICE server の有無にかかわらず gathering し、 host candidate も伝える
  - PATCH body の組み立ては examples/trickle_ice.py の純関数に切り出し、 tests/test_trickle_ice.py で RFC 9725 Figure 3 と突き合わせる
  - なお ICE restart は対象外
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
- [FIX] IceUdpMuxListener の stop() を GIL 解放下で実行するようにする
  - 内部 thread の join が Python callback の GIL 待ちと噛み合って恒停し得るため、 GIL を解放して停止する
  - `IceUdpMuxListener.__del__` からも GIL 解放下で stop() を呼ぶ (Python サブクラスでは破棄時に実行される)
  - 明示 stop() を呼ばずに破棄する経路の恒停には未対応である (破棄時は GIL を保持したまま C++ 側の公開デストラクタが stop() を呼ぶ)
  - @voluntas
- [FIX] WebSocketServer の stop() を GIL 解放下で実行するようにする
  - 受け入れ thread が Python callback の GIL を待っている間に stop() が GIL を保持したまま走ると恒停し得るため、 GIL を解放して停止する
  - `WebSocketServer.__del__` からも GIL 解放下で stop() を呼ぶ (Python サブクラスでは破棄時に実行される)
  - 明示 stop() を呼ばずに破棄する経路の恒停には未対応である (破棄時は GIL を保持したまま C++ 側の公開デストラクタが stop() を呼ぶ)
  - @voluntas
- [FIX] Free-Threading 対応の Python かどうかを判定して FREE_THREADED を明示的に指定する
  - nanobind は GIL ありの Python で `FREE_THREADED` をエラーも警告もなく無効化するため、 指定と実際の ABI が黙って食い違っていた
  - Free-Threading 版の wheel は 3.14t、 GIL あり版は 3.12 / 3.13 / 3.14 で配布する (3.13t 向けは配布しない)
  - ABI から見た判定と nanobind の判定が食い違った場合は CMake の警告で検出する
  - @voluntas
- [FIX] Track.request_keyframe() / Track.request_bitrate() が GIL を保持したまま送信経路に入る問題を修正する
  - 送信経路の内部ロックを保持したまま GIL 待ちに入ると、 受信経路の Python callback と循環待ちになって復帰不能になる (`send` 系と同じ構造)
  - `send` 系と同じく GIL を解放して実行する
  - @voluntas
- [FIX] DependencyDescriptorWriter が context の破棄後に壊れた値を読む問題を修正する
  - writer は context 自体ではなく context のメンバ (structure / descriptor) への参照を保持するため、 context を生存させるようにする
  - 実測: 一時的な context を渡すと、 context の破棄後に `get_size_bits()` が `RuntimeError` になっていた (use-after-free のため結果は不定)
  - @voluntas
- [FIX] PeerConnection.config() の戻り値が親の寿命に紐付かない問題を修正する
  - 親を先に破棄してから戻り値を使うと use-after-free で落ちていた (実測: exit 139)
  - `config()` の戻り値が親を生存させるようにする (`Description.media()` / `Description.application()` の分は `[CHANGE] Description.media() / Description.application() が値 (コピー) を返すようにする` のエントリを参照)
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

- [FIX] mbedTLS をスレッドセーフにビルドする
  - スレッド対応が無い設定でビルドしていたため、 複数の thread から TLS を初期化したときに mbedTLS 内部 (PSA / entropy) の状態が壊れ、 TLS の初期化中に SIGTRAP (malloc の freelist 検査) や SIGSEGV でプロセスが落ちていた
  - WebSocketServer / WebSocket の TLS 接続が稀にプロセスごと落ちる問題を修正する (mbedTLS のビルドを再現可能にするため、 libdatachannel も同じ設定で再ビルドが必要)
  - @voluntas

- [FIX] RtpPacketizer の max_fragment_size に小さい値を渡すとハングし、 メモリを消費し続ける問題を修正する
  - H264RtpPacketizer / H265RtpPacketizer / AV1RtpPacketizer で、 ハングや範囲外アクセスになる max_fragment_size を構築時に拒否する (H264 は 4〜65535、 H265 は 6〜65535、 AV1 は 2 以上。 範囲外は `ValueError` になる)
  - 上限があるのは、 フラグメント長を uint16_t に切り詰める処理があり、 65536 以上では切り詰めで長さが 0 や 1 になるためである
  - `outgoing` を GIL 解放下で実行し、 呼び出し中も他の thread が動けるようにする
  - なお、 AV1 で SequenceHeader をキャッシュした後は max_fragment_size が 2 + SequenceHeader 長 未満だとヒープを壊す経路が残る。 SequenceHeader のキャッシュの有無は binding から判定できないため、 根本解消は libdatachannel 側の修正が必要である
  - @voluntas

- [FIX] 未対応の Windows を PyPI の classifiers と CMakeLists.txt の表明から外す
  - README は未対応 (優先実装) としているため、 表明を README に合わせる
  - @voluntas

- [FIX] examples/whip.py と whep.py で共有する handle_error が structlog の logger に存在しない isEnabledFor を呼び AttributeError になる問題を修正する
  - レベル判定をやめ、 スタックトレースの出力は logger.debug に任せる
  - @voluntas

- [FIX] 生成される型スタブに __init__.py の 6 つのエイリアスを含める
  - 型チェッカーは __init__.py より __init__.pyi を優先するため、 スタブ側に無いと利用者の import が型チェックで失敗していた
  - @voluntas

- [FIX] Candidate の __eq__ と __hash__ の不整合を修正する
  - candidate 行で比較し、 同じ candidate 行なら同一 hash になるようにする (dict / set で畳み込まれなかった)
  - libdatachannel の operator!= は foundation のみを比較しており == と非対称だったため、 __ne__ のバインドを外す
  - @voluntas

- [FIX] DataChannel.close() と Track.close() を GIL 解放下で実行するようにする
  - close() は送信経路や callback と同じ mutex を取るため、 GIL を保持したまま呼ぶと循環待ちになり得る
  - @voluntas

### misc

- [FIX] CI の wheel テスト環境に structlog を追加する
  - tests/test_error_logging.py が examples/error_logging.py を読み込むため、 test グループだけを入れる CI で collection が失敗していた
  - @voluntas

- [FIX] mbedTLS のスレッド対応が CI で有効にならない問題を修正する
  - `CMakeLists.txt` の `_deps` を捨てる条件が、 無効化されたままの行 (//#define) に誤マッチしていた
  - `wheel.yml` と `prek.yml` の `_deps` キャッシュキーに世代を付け、 修正前のキャッシュを復元させない
  - @voluntas

- [FIX] CI の pytest リトライを削除する
  - リトライに頼らず、 pytest の失敗をそのまま job の失敗として扱う (mbedTLS のスレッド対応で不安定要因を解消したため)
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
- [FIX] IceUdpMuxListener の callback で例外を投げてもプロセスが落ちないようにする
  - callback は libjuice の C callback から直接呼ばれるため、 Python の例外が C のフレームを横断すると std::terminate になっていた (実測: exit 134)
  - binding 側で受け止めて RuntimeWarning として記録し、 STUN の未処理 request を送る実測テストを追加する
- [FIX] examples/whip.py の RTP timestamp が長時間の配信で wrap せず、 送信が停止する問題を修正する
  - 毎フレームの差分の足し込みをやめ、 最初の dts からの経過時間から計算する (映像は 90000、 音声は 48000)
  - 乱数の初期値を維持したまま 32 bit で wrap させ、 丸め誤差の累積も解消する
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
