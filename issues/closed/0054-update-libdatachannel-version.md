# libdatachannel を v0.24.2 に上げる

- Priority: Medium
- Created: 2026-10-11
- Completed: 2026-10-11
- Branch: feature/update-libdatachannel-version
- Polished: 2026-10-11

## 目的

deps.json の libdatachannel を v0.24.0 (2025-11-17 公開) から v0.24.2 (2026-03-29 公開) に上げる。 本パッケージを利用する環境が採用しているバージョンに揃え、 v0.24.1 / v0.24.2 に入っているクラッシュとメモリ安全の修正を取り込む。

## 優先度根拠

- v0.24.0 は 2025-11-17 公開で、 上流は v0.24.6 (2026-09-26) まで進んでいる。 本パッケージを利用する環境が採用しているバージョンに揃えることで、 検証の再現性を確保する
- v0.24.1 に WebSocket を onError callback 内で削除したときのクラッシュ修正と、 temporal unit delimiter を持たない AV1 フレームの RTP packetizer 修正が入っている
- v0.24.2 に RTP / RTCP パケット処理の size check 追加、 SctpTransport::sendReset のバッファアラインメント修正、 Description の複数 group 属性の修正が入っている
- v0.24.3 以降にも TLS / DTLS とメモリ安全の修正 (DTLS の受信キュー溢れ、 IceTransport の use-after-free、 DTLS のハンドシェイク恒停など) が入っているため、 追従は別途判断する (本 issue では v0.24.2 までとする)
- 公開 API の削除や破壊的な変更は無い (Description::sessionId と SetThreadPoolSize の追加、 Preload / Cleanup 宣言の並べ替え、 RtcpSr / RtcpRr の getReportBlock が範囲外で nullptr を返す修正とそのコメント追記。 getReportBlock は binding していないため影響なし)

## 実測結果 (2026-10-11)

deps.json の libdatachannel.tag を v0.24.2 に変えて `uv build --wheel` を実行し、 生成した wheel を fresh な venv に install してテストを実行した。

- ビルドは成功した (ビルドオプションの変更は不要)
- 176 passed / 12 skipped / 2 deselected (v0.24.0 と同一の結果)

## 現状

- deps.json の libdatachannel.tag は v0.24.0。 2025.1.0 / 2025.1.1 / 2025.1.2 と 2026.1.0 の開発版はすべて v0.24.0 でリリースしている
- CMakeLists.txt は deps.json の tag を ExternalProject で取得してソースからビルドする (BUILD_SHARED_LIBS=OFF / USE_MBEDTLS=ON / USE_NICE=OFF / NO_TESTS=ON / NO_EXAMPLES=ON)
- ビルドディレクトリは _deps/libdatachannel/<tag>/ で tag ごとに分かれており、 CI のキャッシュキーも deps.json のハッシュを含むため、 tag の変更で自動的に再ビルドされる
- CHANGES.md の `## develop` に「AV1 で SequenceHeader をキャッシュした後は max_fragment_size が 2 + SequenceHeader 長 未満だとヒープを壊す経路が残る」という未解消の記載がある。 v0.24.1 の AV1 packetizer 修正との関連は本 issue の範囲外とする

## 設計方針

- deps.json の libdatachannel.tag を v0.24.2 に変更する (url は変更しない)
- CMakeLists.txt は変更しない (v0.24.2 でも同じビルドオプションが使えることを確認する)
- THIRD_PARTY_LICENSES.md は変更しない (libdatachannel と、 同梱される libjuice / libsrtp / usrsctp のバージョンとライセンス本文が v0.24.0 と同じであることを確認済み)
- 上流で追加された API (SetThreadPoolSize / Description::sessionId) の binding 追加は本 issue の範囲外とする

## 完了条件

- deps.json の libdatachannel.tag が v0.24.2 になっている
- `uv build --wheel` が成功する
- `make develop` の後に CI と同じ 2 件 (`test_destruct_without_explicit_close` と `test_track`) を deselect した pytest が全件 PASS する
- prek のフック (ruff / ty / check-toml / check-yaml / clang-format) が PASS する
- CI (wheel.yml の 24 leg と prek.yml) が PASS する
- CHANGES.md の `## develop` に `[UPDATE] libdatachannel のバージョンを v0.24.2 に上げる` と担当者行 (`- @voluntas`) が追記されている
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること

## 参考

- libdatachannel v0.24.1: https://github.com/paullouisageneau/libdatachannel/releases/tag/v0.24.1
- libdatachannel v0.24.2: https://github.com/paullouisageneau/libdatachannel/releases/tag/v0.24.2
- 対象: deps.json の libdatachannel

## 解決方法

- `deps.json`
  - `libdatachannel.tag` を `v0.24.0` から `v0.24.2` に変更した (`url` は変更なし)
- `CHANGES.md`
  - `## develop` の `[UPDATE]` に `libdatachannel のバージョンを v0.24.2 に上げる` を追記した
- `src/bind_libdatachannel.cpp`
  - `as_audio` の binding にあるコメントの `v0.24.0 の description.cpp` を `上流の description.cpp` に直した。 上流の `createEntry` / `addMedia` は v0.24.0 と v0.24.2 で変わっておらずコメントの前提は成り立つが、 使わなくなったバージョンへの参照を残さないため
- `CMakeLists.txt` / `THIRD_PARTY_LICENSES.md` / README は変更していない (ビルドオプションはそのまま使え、 libdatachannel と同梱される libjuice / libsrtp / usrsctp のバージョンとライセンス本文は v0.24.0 と同じで、 README にバージョン記載が無い)
- 検証
  - `make develop` が成功し、 インストールされた拡張モジュールが `_deps/libdatachannel/v0.24.2/` のソースでビルドされていることを実測で確認した
  - `make develop` の後に `.venv/bin/python -m pytest tests/ -q --timeout=600 --deselect tests/test_peerconnection.py::test_destruct_without_explicit_close --deselect tests/test_peerconnection.py::test_track` を実行し、 176 passed / 12 skipped / 2 deselected だった (`make test` は deselect を渡さず、 `uv run` は sync で `make develop` が入れた拡張モジュールを消す場合があるため使わない)
  - `prek run --all-files ruff-check` が Passed
  - CI (wheel.yml の 24 leg と prek.yml) は PR で確認する
