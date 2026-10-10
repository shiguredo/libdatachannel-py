# THIRD_PARTY_LICENSES.md に nanobind と tsl::robin_map が欠落している

- Priority: Medium
- Created: 2026-08-30
- Completed: {YYYY-MM-DD}
- Branch: feature/update-missing-third-party-licenses
- Polished: 2026-10-10

## 目的

wheel の .so には nanobind (BSD-3-Clause) と、nanobind に同梱され同じくリンクされる tsl::robin_map (MIT) が静的リンクされるが、THIRD_PARTY_LICENSES.md に両者の記載がなく、ライセンス表記義務を満たしていない。

## 優先度根拠

- 静的リンクしたライブラリのライセンス文は再配布時に添付する必要がある (BSD-3-Clause は著作権表示保持条項を含む)
- 配布物のライセンス遵守の問題であり、次回リリース前に解消すべき

## 現状

- THIRD_PARTY_LICENSES.md は libdatachannel / mbedtls / usrsctp / plog / libjuice / libsrtp / nlohmann/json の 7 件を網羅しているが、nanobind と tsl::robin_map が欠落している
- `_build` 配下に `libnanobind-static.a` が存在し、静的リンクが確認できる

## 設計方針

- `THIRD_PARTY_LICENSES.md` に nanobind と tsl::robin_map の節を、 既存の書式 (```text のフェンス、 上流の URL) に合わせて追記する。 本文は次の一次資料から**そのまま転記**する
  - nanobind: `site-packages/nanobind-<version>.dist-info/licenses/LICENSE` (BSD-3-Clause)。 `pyproject.toml` は `nanobind>=3.1.0` で版を固定していないため、 転記時点の版を確認する (確認時点では 3.1.0)
  - tsl::robin_map: nanobind が同梱する `nanobind/ext/robin_map/include/tsl/robin_map.h` の先頭コメント (MIT)。 nanobind の wheel は robin_map の LICENSE ファイルを同梱しないため、 ヘッダのコメントを原文とする
- `pyproject.toml` の `[project] license-files` に `THIRD_PARTY_LICENSES.md` を追加し、 wheel に同梱されるようにする。 現状は `LICENSE` のみで、 `THIRD_PARTY_LICENSES.md` は wheel に入っていない (実測: wheel の中身は `libdatachannel/*` 4 件と `dist-info/{METADATA,RECORD,WHEEL,licenses/LICENSE}` の 8 件のみ)
- 追記位置は既存の節の末尾とする

## 完了条件

- `THIRD_PARTY_LICENSES.md` に nanobind と tsl::robin_map の節があり、 本文が上記の一次資料と一致すること
- 既存の 7 件 (libdatachannel / mbedtls / usrsctp / plog / libjuice / libsrtp / nlohmann-json) と重複していないこと
- `uv build --wheel` で作った wheel に `dist-info/licenses/THIRD_PARTY_LICENSES.md` が含まれ、 `METADATA` に `License-File: THIRD_PARTY_LICENSES.md` があること
- `prek run --all-files pytest` と `prek run --all-files ty` が PASS すること
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること
- `CHANGES.md` への記載は不要 (`.md` のみの変更のため)

## 参考

- 対象: THIRD_PARTY_LICENSES.md
- nanobind (BSD-3-Clause) と nanobind 同梱の tsl::robin_map (MIT) は wheel の .so に静的リンクされる
