# THIRD_PARTY_LICENSES.md に nanobind と tsl::robin_map が欠落している

- Priority: Medium
- Created: 2026-08-30
- Completed: {YYYY-MM-DD}
- Branch: feature/fix-add-missing-third-party-licenses
- Polished: {YYYY-MM-DD}

## 目的

wheel の .so には nanobind (BSD-3-Clause) と、nanobind に同梱され同じくリンクされる tsl::robin_map (MIT) が静的リンクされるが、THIRD_PARTY_LICENSES.md に両者の記載がなく、ライセンス表記義務を満たしていない。

## 優先度根拠

- 静的リンクしたライブラリのライセンス文は再配布時に添付する必要がある (BSD-3-Clause は著作権表示保持条項を含む)
- 配布物のライセンス遵守の問題であり、次回リリース前に解消すべき

## 現状

- THIRD_PARTY_LICENSES.md は libdatachannel / mbedtls / usrsctp / plog / libjuice / libsrtp / nlohmann/json の 7 件を網羅しているが、nanobind と tsl::robin_map が欠落している
- `_build` 配下に `libnanobind-static.a` が存在し、静的リンクが確認できる

## 設計方針

- nanobind と tsl::robin_map のライセンス全文と著作権表記を THIRD_PARTY_LICENSES.md に追記する
- 追加したライセンス文が各リポジトリの原文と一致していることを確認する

## 完了条件

- nanobind と tsl::robin_map のライセンス全文が THIRD_PARTY_LICENSES.md に記載されていること
- 記載内容が各リポジトリの原文と一致していること

## 参考

- 対象: THIRD_PARTY_LICENSES.md
- nanobind (BSD-3-Clause) と nanobind 同梱の tsl::robin_map (MIT) は wheel の .so に静的リンクされる
