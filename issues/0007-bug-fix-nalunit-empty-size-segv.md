# NalUnit と H265NalUnit にヘッダサイズ未満のバッファを渡すと SIGSEGV する

- Priority: High
- Created: 2026-08-30
- Completed: {YYYY-MM-DD}
- Branch: feature/fix-nalunit-empty-size-segv
- Polished: 2026-10-08

## 目的

NalUnit / H265NalUnit のコンストラクタ (size 版 / bytes 版) は、 ヘッダサイズ未満のバッファでも生成できてしまう。 その状態でヘッダアクセスや `set_payload()` を呼ぶと、 0 バイトのバッファでは null ポインタのデリファレンスで SIGSEGV し、 H265NalUnit の 1 バイトのバッファでは 2 バイトのヘッダを読む範囲外アクセスになる。 Python から 1 行で crash する経路を binding 側の入力バリデーションで塞ぐ。

## 優先度根拠

- `NalUnit(0).forbidden_bit()` / `NalUnit(b"").forbidden_bit()` だけで SIGSEGV する (実測)
- H265NalUnit では 1 バイトのバッファでも生成でき、 2 バイトのヘッダを読み書きする範囲外アクセスになる (実測では落ちないが未定義動作)
- Release ビルドでは assert が消えるため、libdatachannel 本体の防御が機能しない
- H264 / H265 の NAL 解析は本ライブラリの主要ユースケースであり、入力バリデーションが binding に欠けている

## 現状

再現手順:

```python
from libdatachannel import NalUnit

n = NalUnit(0)
n.forbidden_bit()  # SIGSEGV
```

- `bind_nalunit` / `bind_h265nalunit` のコンストラクタ binding はヘッダサイズを検証していない
  - `NalUnit`: `nb::init<size_t, bool, NalUnit::Type>()` と `nb::init<binary&&>()` (bytes 版)
  - `H265NalUnit`: `nb::init<size_t, bool>()` と `nb::init<binary&&>()`
- libdatachannel の `NalUnit(size_t, bool, Type)` ctor (`include/rtc/nalunit.hpp`) は `including_header=true` のとき size を加算しないため、 size 0 のバッファがそのまま作られる
- `NalUnit::header()` は `assert(size() >= 1)`、 `H265NalUnit::header()` は `assert(size() >= 2)` でしかガードされておらず、 Release ビルド (CMakeLists.txt が libdatachannel を `-DCMAKE_BUILD_TYPE=Release` でビルド) では assert が消滅する
- 実測 (SIGSEGV):
  - `NalUnit(0).forbidden_bit()` / `H265NalUnit(0).forbidden_bit()` / `NalUnit(0, True, NalUnit.Type.H265).forbidden_bit()`
  - `NalUnit(b"").forbidden_bit()` / `H265NalUnit(b"").forbidden_bit()` / `NalUnit(b"").set_payload(b"a")` (bytes 版も同じ経路)
  - `NalUnit(0).set_payload(b"a")` / `H265NalUnit(0).set_payload(b"a")` (`set_payload()` は `assert` のみで、 0 バイトのバッファに書く)
  - 桁あふれ: `NalUnit(2**64 - 1, False).forbidden_bit()` と `NalUnit(2**64 - 2, False, NalUnit.Type.H265).forbidden_bit()` (確保サイズの加算が wrap して空バッファになる)
- 実測 (SIGSEGV しないが範囲外アクセス): `H265NalUnit(1)` / `H265NalUnit(b"\x00")` は 1 バイトのバッファしか無いが (H265 のヘッダは 2 バイト)、 ヘッダの 2 バイト目を読む `nuh_layer_id()` / `nuh_temp_id_plus1()` と、 書く `set_nuh_layer_id()` / `set_nuh_temp_id_plus1()` が範囲外を読み書きする。 1 バイト目しか読まない `forbidden_bit()` / `unit_type()` は落ちない
- 実測 (SIGSEGV しない): `NalUnit(0, False)` / `NalUnit(0, False, NalUnit.Type.H265)` / `H265NalUnit(0, False)` は ctor がヘッダサイズ (H264 は +1、 H265 は +2) を加算するため 1 バイト以上が確保される。 `NalUnit(0, False)` は `NalUnit()` と等価
- `payload()` は範囲外で SIGSEGV せず `ValueError` (libc++ の range ctor) になる。 SIGSEGV するのは `set_payload()` 側

## 設計方針

- binding 側で「確保されるバッファがヘッダサイズ以上か」を検証し、 範囲外は `nb::value_error` を投げる
  - `nb::init<size_t, bool, NalUnit::Type>()`: `including_header=true` のとき `size >= 1` (H264 のヘッダサイズ)。 `NalUnit` の `type` に `H265` を渡しても `NalUnit::header()` は 1 バイトしか読まないため、 必要な下限は 1 バイトとする (実測で `NalUnit(1, True, NalUnit.Type.H265)` は落ちない)
  - `nb::init<size_t, bool>` (`H265NalUnit`): `including_header=true` のとき `size >= 2` (H265 のヘッダサイズ)
  - `nb::init<binary&&>()`: `len(data) >= ヘッダサイズ` (H264 は 1、 H265 は 2)
  - `including_header=false`: 確保サイズは `size + ヘッダサイズ` になるため、 桁あふれでヘッダサイズを下回る場合 (`size > SIZE_MAX - ヘッダサイズ`) を拒否する。 size 0 は確保サイズがヘッダサイズと等しいので拒否しない
- コンストラクタで下限を保証すれば、 ヘッダアクセスと `set_payload()` の SIGSEGV と範囲外アクセスは解消する (メソッド側の追加ガードは不要)
- 実測した再現ケース (SIGSEGV / 範囲外アクセス / 桁あふれ) と、 従来どおり成功する `including_header=false` の size 0 を回帰テストとして `tests/test_nalunit.py` と `tests/test_h265nalunit.py` に追加する

## 完了条件

- ヘッダサイズ未満のバッファが確保される生成が `ValueError` になること。 対象は `NalUnit(0)` / `NalUnit(0, True, NalUnit.Type.H265)` / `H265NalUnit(0)` / `H265NalUnit(1)` / `NalUnit(b"")` / `H265NalUnit(b"")` / `H265NalUnit(b"\x00")` と、 桁あふれする size
- 生成が例外になった後もプロセスが落ちないこと (SIGSEGV しないこと)。 `set_payload()` を呼ぶ経路も含めて確認する
- `including_header=false` で size 0 の生成は従来どおり成功すること (`NalUnit(0, False)` / `NalUnit(0, False, NalUnit.Type.H265)` / `H265NalUnit(0, False)`)
- H264 / H265 / `including_header` / size 版 / bytes 版の組み合わせをカバーするテストが `tests/test_nalunit.py` と `tests/test_h265nalunit.py` に追加されていること
- `uv sync && make test` で全テストが PASS すること (既知の恒停を持つテストは [[0005-bug-fix-destructor-callback-deadlock]] の対象)
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること

## 参考

- 対象シンボル: `bind_nalunit`、`bind_h265nalunit` (src/bind_libdatachannel.cpp)
- libdatachannel v0.24.0: `include/rtc/nalunit.hpp` (`NalUnit::NalUnit(size_t, bool, Type)`、`NalUnit::header()`、`NalUnit::setPayload()`)、 `include/rtc/h265nalunit.hpp`
- 関連 issue: [[0006-bug-fix-send-size-out-of-bounds-read]] (binding 側の size 検証。 同じ `nb::value_error` を使う)、 [[0026-test-add-missing-binding-tests]] (未テスト binding の洗い出し)
