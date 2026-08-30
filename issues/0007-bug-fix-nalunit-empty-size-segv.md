# NalUnit と H265NalUnit に size 0 を渡すと SEGV する

- Priority: High
- Created: 2026-08-30
- Completed: {YYYY-MM-DD}
- Branch: feature/fix-nalunit-empty-size-segv
- Polished: {YYYY-MM-DD}

## 目的

NalUnit / H265NalUnit の size 引数に 0 を渡すと、ヘッダアクセスメソッドの呼び出し時に null ポインタのデリファレンスが発生し、プロセスが落ちる。Python から 1 行で crash する経路を binding 側の入力バリデーションで塞ぐ。

## 優先度根拠

- `NalUnit(0).forbidden_bit()` だけで SIGSEGV する (実測)
- Release ビルドでは assert が消えるため、libdatachannel 本体の防御が機能しない
- H264 / H265 の NAL 解析は本ライブラリの主要ユースケースであり、入力バリデーションが binding に欠けている

## 現状

再現手順:

```python
from libdatachannel import NalUnit

n = NalUnit(0)
n.forbidden_bit()  # SIGSEGV
```

- `bind_nalunit` / `bind_h265nalunit` は `nb::init<size_t, ...>` をそのままバインドし、size 0 を許す
- libdatachannel の `NalUnit(size_t, bool, Type)` ctor (`include/rtc/nalunit.hpp`) は `including_header=true` のとき size を加算しないため、size 0 のバッファがそのまま作られる
- `NalUnit::header()` は `assert(size() >= 1)` でしかガードされておらず、Release ビルド (CMakeLists.txt が libdatachannel を `-DCMAKE_BUILD_TYPE=Release` でビルド) では assert が消滅する
- 実測: `NalUnit(0).forbidden_bit()` / `H265NalUnit(0).forbidden_bit()` / `NalUnit(0, True, NalUnit.Type.H265).forbidden_bit()` のすべてが SIGSEGV
- `including_header=False` のときはヘッダサイズ (H264 は +1、H265 は +2) が加算されるため、size 0 でも確保は発生する
- `payload()` / `set_payload()` も assert のみのガードで、同様の経路がある

## 設計方針

- binding 側で size を検証する。`including_header=True` の場合、H264 は `size >= 1`、H265 は `size >= 2` を要求する
- `including_header=False` の場合も、加算後の確保サイズが最低ヘッダサイズを満たすかを検証する
- 範囲外のときは `nb::value_error` を投げる
- 実測した再現ケースを回帰テストとして追加する

## 完了条件

- size 0 での生成とヘッダアクセスが例外になること (SEGV しないこと)
- H264 / H265 / `including_header` の組み合わせをカバーするテストが追加されていること
- `uv sync && make test` で全テストが PASS すること
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること

## 参考

- 対象シンボル: `bind_nalunit`、`bind_h265nalunit` (src/bind_libdatachannel.cpp)
- libdatachannel v0.24.0: `include/rtc/nalunit.hpp` (`NalUnit::NalUnit(size_t, bool, Type)`、`NalUnit::header()`)、`include/rtc/h265nalunit.hpp`
