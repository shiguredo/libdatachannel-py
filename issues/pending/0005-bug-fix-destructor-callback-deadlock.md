# PeerConnection destructor 経路で同期 callback が引き起こす hang の根本対応

- Priority: Low
- Created: 2026-05-18
- Model: Opus 4.7
- Branch: 未定 (設計判断後に決定する)

## pending とした理由

本 issue は **当面着手しない** ため `issues/pending/` 配下に置く。 理由は以下のとおり:

- 根本対応は複数の設計選択肢 (案 G / H / I / J) があり、 どれを採るかは **nanobind の API 仕様調査** や **libdatachannel upstream の方針確認** といった事前調査を伴う。
- 既に [[0001-bug-fix-peer-connection-destructor-gil-release]] / [[0002-bug-fix-websocket-destructor-gil-release]] / [[0003-bug-fix-websocketserver-destructor-gil-release]] / [[0004-bug-fix-ice-udp-mux-listener-destructor-gil-release]] で症状緩和の基本対策が入る前提があり、 通常利用 (libdatachannel callback 未登録 / callback で I/O block しない / 明示 `close()` を呼ぶ) では実用上 hang まで至らない。
- リソース投下の優先度は他タスクに譲るが、 構造的問題を忘れないために issue として残す。

## 目的

`PeerConnection` を destruct する経路で、 libdatachannel 本体の以下の構造に起因する hang を根本的に解消する。

- `synchronized_callback::operator()` が **mutex を保持したまま callback を同期実行する** (`_deps/libdatachannel/v0.24.0/source/include/rtc/utils.hpp:79`)
- `changeState(Closed)` / `changeIceState(Closed)` が **callback を同期実行する** (`_deps/libdatachannel/v0.24.0/source/src/impl/peerconnection.cpp:1321`, `:1339`)
- `~PeerConnection()` (impl) の `mProcessor.join()` が **public destructor 内で呼ばれる** (= GIL 保持中、 `_deps/libdatachannel/v0.24.0/source/src/impl/peerconnection.cpp:82`)

これらが組み合わさると、 callback が Python 関数を呼んで GIL 待ち / I/O block すると、 state 遷移と `mProcessor.join()` が完了せず、 main thread の close 完了待ちが永続化する。 [[0001-bug-fix-peer-connection-destructor-gil-release]] では「GIL release + polling timeout + wrapper」 で *症状緩和* しているが、 「callback が I/O block する条件下で 30 秒 timeout まで待たされる」 という事象自体は残る。 本 issue ではこれを構造から取り除く。

## 優先度根拠

- [[0001-bug-fix-peer-connection-destructor-gil-release]] で症状緩和済みのため、 通常利用シナリオでの実害は低い。
- 残るのは「callback が I/O block する条件 (典型: pytest stdout capture + `print` callback + `Track.on_closed` 等) で 30 秒の timeout 待ちが発生する」 シナリオ。 これはテスト環境固有のケースが多く、 本番利用での顕在化頻度は低い。
- 解決には binding 側の大規模変更 (案 G/J) か外部依存への PR (案 H) か未調査の API 利用 (案 I) が必要で、 投資対効果が読みにくい。
- 以上から Priority: Low とし、 当面 pending。

## 現状

[[0001-bug-fix-peer-connection-destructor-gil-release]] が入った状態を前提として、 以下が残る。

- `~PeerConnection()` 内 `mProcessor.join()` は依然として GIL 保持下で実行される (binding の destructor hook には触れていない)。
- `changeState(Closed)` 内の callback 同期実行も libdatachannel 本体側のため変わらない。
- 結果として、 callback で I/O block する条件下では `close()` の polling が 30 秒 timeout まで待たされ、 `RuntimeWarning` が出る。 利用者から見ると「30 秒固まる」 ように見える。

実証結果 (サマリ抜粋):

| 環境 | callback で `print` | 結果 |
|---|---|---|
| 通常 python 直接 (`uv run python ...`) | あり | PASS |
| 通常 python 直接 | なし | PASS |
| pytest デフォルト capture | あり | hang |
| pytest デフォルト capture | なし | PASS (1.11 秒) |
| pytest `-s` 付き | あり | PASS (2.10 秒) |

加えて、 ユーザー検証では「直接実行でも callback の `print` + `Track.on_closed` の組み合わせで hang する場合がある」。 つまり pytest stdout capture 単独の問題ではなく、 callback I/O + `mProcessor.join()` の GIL 待ちが真因。

## 設計方針

以下の案 (サマリ「検討した修正案と結果」 由来) のいずれか、 もしくは組み合わせで対応する。 採用案は事前調査の結果で決める。

### 案 G: callback wrapper で Python 呼び出しを非同期化する (fire-and-forget)

- binding 側の callback wrapper で Python 関数呼び出しを別 thread に detach する。
- libdatachannel 内の mutex は即時 release されるため、 `synchronized_callback` 由来の deadlock を回避できる。
- 副作用:
  - callback の **順序保証が壊れる** (state change が逆順に届くなど)
  - lifetime 管理が複雑 (detach 先の thread が PyObject を保持する期間)
  - 例外伝播が困難 (detach 先の例外をどう扱うか)
- 影響範囲: 大規模 (全 callback 系の binding を見直す)

### 案 H: libdatachannel upstream に PR

- `changeState` 等の同期 callback 実行を非同期化する (or オプション化する) 提案を upstream に出す。
- 影響範囲: 外部依存。 採否・リリース時期がコントロールできない。
- 工数: 大 (設計提案・実装・レビュー対応)

### 案 I: nanobind の `tp_dealloc` フック等で GIL release 状態の C++ destructor を実現する

- nanobind の destructor hook (調査要) を使って、 Python オブジェクト破棄経路でも C++ destructor を GIL release で走らせる。
- これにより `mProcessor.join()` の GIL 保持問題は緩和される。
- 影響範囲: 中 (destructor hook 周りの binding を追加)。
- 前提: nanobind の公式 doc (https://nanobind.readthedocs.io/en/latest/) で該当 API の有無と仕様を確認すること。 **未調査**。

### 案 J: callback を order-preserving queue に enqueue する wrapper

- callback wrapper で「キューに入れて即時 return」 し、 別 thread が順次取り出して Python 関数を呼ぶ。
- 案 G の弱点である「順序保証が壊れる」 を解消できる。
- 副作用:
  - 戻り値 / 例外 / lifetime 管理が依然複雑
  - キュー処理 thread のライフサイクル管理が必要
- 影響範囲: 大 (案 G と同等)

### 不採用が確認済みの案

- **案 E** (`__del__` で callback を no-op に置換): `synchronized_callback::operator=` が `operator()` と同じ mutex を取るため、 callback 実行中の setter は新たな deadlock を作る。 不採用。
- **案 F** (`__del__` で `reset_callbacks()` を呼ぶ): `changeState(Closed)` は callback を `std::move` してから同期実行するため、 reset 後に nullptr で `std::bad_function_call` を throw するリスク。 安全でない。 不採用。
- **案 D** (callback wrapper の GIL 制御): pipe block は OS レベルの I/O block で GIL とは独立。 効かない。 不採用。

## 完了条件

- 採用案 (G / H / I / J のいずれか) を決定した時点で、 当 issue を `issues/pending/` から `issues/` に戻し、 対応用 issue を新規発行 or 当 issue 自体に実装方針を確定して着手する。
- 実装後の完了条件は以下を想定する。
  - `tests/test_peerconnection.py::test_destruct_without_explicit_close` を **pytest デフォルト capture + callback で `print` + `Track.on_closed` を登録した条件下** でも hang せず、 timeout warning も出さずに PASS する。
  - `gc.collect()` を強制呼びするテストを追加し、 PASS する。
  - 既存テスト全件が PASS する。
  - レビュー観点 (致命的 / 重要) が 0 件であること。

## 解決方法

採用案決定後に確定する。 現時点では未確定。

## 着手判断のトリガ

以下のいずれかが起きたら、 pending 解除を検討する。

- 利用者から「`close()` が timeout warning と共に 30 秒待たされる」 等の報告が増える。
- libdatachannel callback を登録する利用形態が増え、 callback I/O block 由来の hang が顕在化する。
- libdatachannel upstream で同期 callback 構造に変更が入り (案 H が外部から進む)、 案 G/J の負債が軽くなる。
- nanobind 側で destructor hook 周りの API が整理され (案 I)、 投資対効果が改善する。

## 参考

- 仕切り直しサマリ: `/tmp/destructor-gil-release-summary.md`
  - 「未解決の根本問題」 (現象・根本原因・実証結果)
  - 「検討した修正案と結果」 (案 A 〜 J)
- 関連 issue: [[0001-bug-fix-peer-connection-destructor-gil-release]] / [[0002-bug-fix-websocket-destructor-gil-release]] / [[0003-bug-fix-websocketserver-destructor-gil-release]] / [[0004-bug-fix-ice-udp-mux-listener-destructor-gil-release]]
- libdatachannel 関連コード位置:
  - `_deps/libdatachannel/v0.24.0/source/include/rtc/utils.hpp:79` (`synchronized_callback::operator()` の mutex 保持実行)
  - `_deps/libdatachannel/v0.24.0/source/src/impl/peerconnection.cpp:82` (`~PeerConnection()` impl の `mProcessor.join()`)
  - `_deps/libdatachannel/v0.24.0/source/src/impl/peerconnection.cpp:1321-1323` (`changeState(Closed)` の同期 callback 実行)
  - `_deps/libdatachannel/v0.24.0/source/src/impl/peerconnection.cpp:1339-1341` (`changeIceState(Closed)` の同期 callback 実行)
- nanobind 公式ドキュメント (案 I の事前調査用): https://nanobind.readthedocs.io/en/latest/
