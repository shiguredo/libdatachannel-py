# 生成される型スタブに __init__.py の 6 つのエイリアスが欠落する

- Priority: Medium
- Created: 2026-08-30
- Completed: {YYYY-MM-DD}
- Branch: feature/fix-stub-missing-aliases
- Polished: {YYYY-MM-DD}

## 目的

src/libdatachannel/__init__.py が定義する AACRtpPacketizer 等 6 つの公開エイリアスが、nanobind_add_stub が生成する __init__.pyi に存在しない。型チェッカーは __init__.pyi を __init__.py より優先するため、wheel 利用者の `from libdatachannel import AACRtpPacketizer` が型チェックで失敗する (実行時は成功するため、気づきにくい)。

## 優先度根拠

- ty で実証: `from libdatachannel import AACRtpPacketizer` が unresolved-import になる
- 公開 API (エイリアス) と配布するスタブの乖離は、利用者コードの型チェックを壊す

## 現状

再現手順:

```python
from libdatachannel import AACRtpPacketizer  # 実行時は成功、型チェックは失敗
```

- `src/libdatachannel/__init__.py` は AACRtpPacketizer / PCMURtpPacketizer / G722RtpPacketizer / AACRtpDepacketizer / PCMURtpDepacketizer / G722RtpDepacketizer の 6 つのエイリアスを定義する
- `nanobind_add_stub` (CMakeLists.txt) は extension module のみのスタブを生成し、Makefile の develop ターゲットが `_build/__init__.pyi` を `src/libdatachannel/` にコピーする
- スタブ内には Opus / PCMA 系の 4 クラスのみ存在し、エイリアスは 0 件

## 設計方針

- stub 生成後の post-process でエイリアス定義 6 行を `__init__.pyi` に追記する (CMakeLists.txt の stub 生成後に実行する形)
- または stub を `libdatachannel_ext.pyi` として生成し、手書きの `__init__.pyi` (`from .libdatachannel_ext import *` + エイリアス) を配置する構成に変える
- どちらを採用するかは、wheel に含まれるスタブの構成を確認して決める

## 完了条件

- wheel 内のスタブで 6 つのエイリアスが解決すること (ty で確認)
- 生成フローの整合が保たれていること (make wheel で壊れないこと)
- `uv sync && make test` で全テストが PASS すること
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること

## 参考

- 対象: src/libdatachannel/__init__.py、CMakeLists.txt (`nanobind_add_stub`)、Makefile (`develop` ターゲット)
