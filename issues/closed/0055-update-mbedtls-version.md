# mbedTLS を v3.6.7 に上げる

- Priority: Medium
- Created: 2026-10-11
- Completed: 2026-10-11
- Branch: feature/update-mbedtls-version
- Polished: 2026-10-11

## 目的

deps.json の mbedtls を v3.6.5 から v3.6.7 に上げる。 3.6 系は LTS で 2027-03 まで bug fix と security fix が提供されると宣言されており、 v3.6.6 / v3.6.7 に入っているセキュリティ修正を取り込む。

## 優先度根拠

- v3.6.6 (2026-03-31) と v3.6.7 (2026-07-07) はセキュリティ修正を含む。 代表例は TLS 1.3 のセッション再開時のなりすまし (v3.6.6)、 DTLS の early renegotiation 時のヒープ破壊 (v3.6.7)、 TLS 1.2 ECDHE-PSK のリモートバッファオーバーフロー (v3.6.7)、 mbedtls_pkcs7_free の use-after-free (v3.6.7)、 X.509 の basicConstraints 検証不備 (v3.6.7)
- mbedTLS は wheel に静的リンクされて配布されるため、 修正はパッケージの更新でしか利用者に届かない
- 現在の pin である v3.6.5 は 2026.1.0 の開発中に 3.6.4 から上げたもので、 その後 2 つの patch が出ている
- 3.6 系は LTS (2027-03 まで) のため、 4 系への更新を待たずにセキュリティ修正だけを取り込める
- 4 系 (4.1.0 / 4.1.1 / 4.2.0) は libdatachannel が `find_package(MbedTLS 3 REQUIRED)` で 3 系を要求するため対象外とする

## 実測結果 (2026-10-11)

deps.json の mbedtls.tag を v3.6.7 に変えて `uv build --wheel` を実行し、 生成した wheel を fresh な venv に install してテストを実行した。

- ビルドは成功し、 `_deps/mbedtls/v3.6.7/install/include/mbedtls/mbedtls_config.h` に MBEDTLS_THREADING_C / MBEDTLS_THREADING_PTHREAD / MBEDTLS_SSL_DTLS_SRTP が定義されていることを確認した
- 176 passed / 12 skipped / 2 deselected (v3.6.5 と同一の結果)

## 現状

- deps.json の mbedtls.tag は v3.6.5
- CMakeLists.txt は mbedTLS を ExternalProject で取得し、 `scripts/config.py` で MBEDTLS_SSL_DTLS_SRTP を、 Windows 以外では MBEDTLS_THREADING_C と MBEDTLS_THREADING_PTHREAD を設定してからビルドする
- mbedTLS のビルドディレクトリは _deps/mbedtls/<tag>/ で tag ごとに分かれる。 tag を変えるとビルド済みライブラリが無いため MBEDTLS_NEEDS_REBUILD が真になり、 _deps/libdatachannel/<libdatachannel の tag>/{build,install} も削除されて libdatachannel が新しい mbedTLS で再ビルドされる (CMakeLists.txt の該当ロジック)
- libdatachannel は mbedTLS のヘッダとスタティックライブラリを MbedTLS_ROOT 経由で使う

## 設計方針

- deps.json の mbedtls.tag を v3.6.7 に変更する (url は変更しない)
- CMakeLists.txt は変更しない (config.py のサブコマンドと threading の設定は 3.6 系で共通。 3.6.7 でも同じ手順が通ることを確認する)
- THIRD_PARTY_LICENSES.md は変更しない (mbedTLS のライセンス本文が v3.6.5 と同じであることを確認済み)
- CHANGES.md の `## develop` には、 2025.1.2 が使っていた 3.6.4 からの最終的な差分を示すエントリを 1 件だけ置く。 中間状態である v3.6.5 のエントリは v3.6.7 のエントリで置き換える
- 4 系への更新は libdatachannel 側の対応が必要なため、 本 issue の範囲外とする

## 完了条件

- deps.json の mbedtls.tag が v3.6.7 になっている
- `uv build --wheel` が成功し、 ビルド時に config.py による MBEDTLS_THREADING_C の設定が成功している
- `make develop` の後に CI と同じ 2 件 (`test_destruct_without_explicit_close` と `test_track`) を deselect した pytest が全件 PASS する
- prek のフック (ruff / ty / check-toml / check-yaml / clang-format) が PASS する
- CI (wheel.yml の 24 leg と prek.yml) が PASS する
- CHANGES.md の `## develop` に `[UPDATE] mbedTLS のバージョンを v3.6.7 に上げる` があり、 中間状態の 3.6.5 のエントリが残っていない
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること

## 参考

- mbedtls 3.6.6: https://github.com/Mbed-TLS/mbedtls/releases/tag/mbedtls-3.6.6
- mbedtls 3.6.7: https://github.com/Mbed-TLS/mbedtls/releases/tag/mbedtls-3.6.7
- 対象: deps.json の mbedtls

## 解決方法

- `deps.json`
  - `mbedtls.tag` を `v3.6.5` から `v3.6.7` に変更した (`url` は変更なし)
- `CHANGES.md`
  - `## develop` の `[UPDATE]` の中間状態だったエントリ (`mbedTLS のバージョンを 3.6.5 に更新する`) を `mbedTLS のバージョンを v3.6.7 に上げる` に置き換えた (2025.1.2 が使っていた 3.6.4 からの最終的な差分を示す 1 件にする)
- `CMakeLists.txt` / `THIRD_PARTY_LICENSES.md` / README は変更していない (config.py のサブコマンドと threading の設定は 3.6 系で共通で、 mbedTLS のライセンス本文は v3.6.5 と同じで、 README にバージョン記載が無い)
- 検証
  - `make develop` が成功し、 `_deps/libdatachannel/v0.24.2/build/CMakeCache.txt` の `MbedTLS_ROOT` が `_deps/mbedtls/v3.6.7/install` を指すことを確認した (libdatachannel が v3.6.7 で再リンクされている)。 ローカルの `_deps/libdatachannel/v0.24.2/{build,install}` は mbedtls v3.6.5 でビルド済みだったため、 削除してから再ビルドさせた
  - `_deps/mbedtls/v3.6.7/install/include/mbedtls/mbedtls_config.h` に MBEDTLS_THREADING_C / MBEDTLS_THREADING_PTHREAD / MBEDTLS_SSL_DTLS_SRTP が定義されていることを確認した
  - インストール済みの拡張モジュールに `mbedtls/v3.6.7` のソースパスが埋め込まれ、 `mbedtls/v3.6.5` の文字列が 0 件であることを確認した (wheel に同梱される mbedTLS が v3.6.7 であることの確認)
  - `make develop` の後に `.venv/bin/python -m pytest tests/ -q --timeout=600 --deselect tests/test_peerconnection.py::test_destruct_without_explicit_close --deselect tests/test_peerconnection.py::test_track` を実行し、 176 passed / 12 skipped / 2 deselected だった
  - `prek run --all-files check-toml` / `check-yaml` と `ruff format --check` / `ruff check` が Passed
  - CI (wheel.yml の 24 leg と prek.yml) は PR で確認する
