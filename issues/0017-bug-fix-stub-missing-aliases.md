# 生成される型スタブに __init__.py の 6 つのエイリアスが欠落する

- Priority: Medium
- Created: 2026-08-30
- Completed: {YYYY-MM-DD}
- Branch: feature/fix-stub-missing-aliases
- Polished: 2026-10-10

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

- `nanobind_add_stub` の OUTPUT を `libdatachannel_ext.pyi` に変え (生成物は extension module のスタブのため)、 その後に `__init__.py` のエイリアスを追記した `__init__.pyi` を作る。 `install(FILES .../__init__.pyi ...)` はこの追記後のファイルを対象にする
- 追記する内容は `src/libdatachannel/_stub_aliases.pyi` としてリポジトリに置き、 `__init__.py` のエイリアスと対で管理する (CMake のスクリプト `cmake/append_stub_aliases.cmake` が連結する)
- 追記は CMake のビルドで行う (`make develop` のコピー手順だけに頼ると、 wheel を作る CI では直らないため)
- 検証は「wheel 内のスタブで 6 つのエイリアスが解決すること」を、 エイリアスを import する `tests/test_stub_aliases.py` を置いて `prek run --all-files ty` で確認する (型チェッカーは `__init__.pyi` を優先するため、 この import が通ること自体が検証になる)。 併せて、 インストールされた `__init__.pyi` にエイリアスの行が含まれることをテストで確認する

## 完了条件

- `tests/test_stub_aliases.py` が通り、 `prek run --all-files ty` で 6 つのエイリアスの import が解決すること
- インストールされた `__init__.pyi` に 6 つのエイリアスが含まれること (テストで確認)
- `make develop` と `make wheel` が成功すること
- `prek run --all-files pytest` と `prek run --all-files ty` が PASS すること (`uv sync` は `make develop` で入れた拡張モジュールを削除するため使わない)
- `CHANGES.md` の `## develop` に `[FIX]` として記録すること
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること

## 参考

- 対象: src/libdatachannel/__init__.py、CMakeLists.txt (`nanobind_add_stub`)、Makefile (`develop` ターゲット)
