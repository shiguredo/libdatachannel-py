# mbedtls を v3.6.7 に上げる

- Priority: Medium
- Created: 2026-10-11
- Completed: {YYYY-MM-DD}
- Branch: feature/update-mbedtls-version
- Polished: {YYYY-MM-DD}

## 目的

deps.json の mbedtls を v3.6.5 から v3.6.7 に上げる。 3.6 系は LTS で 2027-03 まで bug fix と security fix が提供されると宣言されており、 v3.6.6 / v3.6.7 に入っているセキュリティ修正を取り込む。

## 優先度根拠

- v3.6.6 (2026-03-31) と v3.6.7 (2026-07-07) はセキュリティ修正を含む。 代表例は TLS 1.3 のセッション再開時のなりすまし (v3.6.6)、 DTLS の early renegotiation 時のヒープ破壊 (v3.6.7)、 TLS 1.2 ECDHE-PSK のリモートバッファオーバーフロー (v3.6.7)、 mbedtls_pkcs7_free の use-after-free (v3.6.7)、 X.509 の basicConstraints 検証不備 (v3.6.7)
- mbedtls は wheel に静的リンクされて配布されるため、 修正はパッケージの更新でしか利用者に届かない
- 現在の pin である v3.6.5 は 2026.1.0 の開発中に 3.6.4 から上げたもので、 その後 2 つの patch が出ている
- 3.6 系は LTS (2027-03 まで) のため、 4 系への更新を待たずにセキュリティ修正だけを取り込める
- 4 系 (4.1.0 / 4.1.1 / 4.2.0) は libdatachannel が `find_package(MbedTLS 3 REQUIRED)` で 3 系を要求するため対象外とする

## 現状

- deps.json の mbedtls.tag は v3.6.5
- CMakeLists.txt は mbedtls を ExternalProject で取得し、 `scripts/config.py` で MBEDTLS_SSL_DTLS_SRTP を、 Windows 以外では MBEDTLS_THREADING_C と MBEDTLS_THREADING_PTHREAD を設定してからビルドする
- mbedtls のビルドディレクトリは _deps/mbedtls/<tag>/ で tag ごとに分かれる。 tag を変えるとビルド済みライブラリが無いため MBEDTLS_NEEDS_REBUILD が真になり、 _deps/libdatachannel/<libdatachannel の tag>/{build,install} も削除されて libdatachannel が新しい mbedtls で再ビルドされる (CMakeLists.txt の該当ロジック)
- libdatachannel は mbedtls のヘッダとスタティックライブラリを MbedTLS_ROOT 経由で使う

## 設計方針

- deps.json の mbedtls.tag を v3.6.7 に変更する (url は変更しない)
- CMakeLists.txt は変更しない (config.py のサブコマンドと threading の設定は 3.6 系で共通。 3.6.7 で同じ手順が通ることを確認する)
- THIRD_PARTY_LICENSES.md は変更しない (mbedtls のライセンス本文は変わっていない)
- 4 系への更新は libdatachannel 側の対応が必要なため、 本 issue の範囲外とする

## 完了条件

- deps.json の mbedtls.tag が v3.6.7 になっている
- `uv build --wheel` が成功し、 ビルド時に config.py による MBEDTLS_THREADING_C の設定が成功している
- `make develop` の後に CI と同じ 2 件 (`test_destruct_without_explicit_close` と `test_track`) を deselect した pytest が全件 PASS する
- prek のフック (ruff / ty / check-toml / check-yaml / clang-format) が PASS する
- CI (wheel.yml の 24 leg と prek.yml) が PASS する
- CHANGES.md の `## develop` に `[UPDATE] mbedtls のバージョンを v3.6.7 に上げる` が追記されている
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること

## 参考

- mbedtls 3.6.6: https://github.com/Mbed-TLS/mbedtls/releases/tag/mbedtls-3.6.6
- mbedtls 3.6.7: https://github.com/Mbed-TLS/mbedtls/releases/tag/mbedtls-3.6.7
- 対象: deps.json の mbedtls
