# libdatachannel を 0.24 系の最新 (v0.24.6) に追従する

- Priority: Medium
- Created: 2026-10-11
- Completed: {YYYY-MM-DD}
- Branch: feature/update-libdatachannel-latest
- Polished: {YYYY-MM-DD}

## 目的

deps.json の libdatachannel を v0.24.2 から 0.24 系の最新である v0.24.6 (2026-09-26 公開) に上げ、 v0.24.3〜v0.24.6 に入っている TLS / DTLS とメモリ安全の修正を取り込む。

## 優先度根拠

- v0.24.3〜v0.24.6 の修正が本パッケージの構成 (USE_MBEDTLS=ON、 WebSocket / WebSocketServer の TLS、 DTLS、 TURN リレー) に直接関係する
  - v0.24.3: DTLS の受信キューが満杯ならパケットを捨てる (従来はキューが無制限に伸びる)、 DTLS ClientHello と registerIncoming() の競合、 MbedTLS / GnuTLS での WebSocket の idle 切断
  - v0.24.4: MbedTLS の DTLS ReadCallback が実受信バイト数を返す、 IceTransport::RecvCallback の use-after-free
  - v0.24.5: WebSocket 接続の TLS bad record mac
  - v0.24.6: DTLS の起動順序、 TURN リレーでの DTLS ハンドシェイク恒停、 MbedTLS の DTLS timeout によるスレッドプール枯渇、 MbedTLS の証明書 verify callback
- libdatachannel と mbedTLS は wheel に静的リンクされて配布されるため、 修正はパッケージの更新でしか利用者に届かない
- v0.24.2 への追従は 2026.1.0 の開発中に行った (利用環境が採用しているバージョンに揃えるため)。 0.24 系の最新への追従はその次の段階として必要かどうかを判断する必要がある
- v0.24.6 でビルドとテストが通ることは実測済み (下記)

## 実測結果 (2026-10-11)

deps.json の libdatachannel.tag を v0.24.6 に変えて `uv build --wheel` を実行し、 生成した wheel を fresh な venv に install してテストを実行した。

- ビルドは成功した (ビルドオプションの変更は不要)
- 176 passed / 12 skipped / 2 deselected (v0.24.0 / v0.24.2 と同一の結果)

## 現状

- deps.json の libdatachannel.tag は v0.24.2
- CMakeLists.txt は deps.json の tag を ExternalProject で取得してソースからビルドする。 ビルドディレクトリは _deps/libdatachannel/<tag>/ で tag ごとに分かれ、 CI のキャッシュキーも deps.json のハッシュを含むため、 tag の変更で再ビルドされる
- v0.24.6 では同梱される libjuice (v1.7.1 → v1.7.4) と libsrtp (v2.8.0) のバージョンが上がる。 usrsctp は変わらない
- 公開 API の変更は追加と非推奨化が中心で、 binding が使う経路には影響しない (RtcpReportBlock::getNTPOfSR の戻り値が uint32_t から uint64_t に変わるが未 binding。 RtcpRemb の API 追加と setSsrc / getNumSSRC の非推奨化があるが RtcpRemb は未使用。 PacingHandler::run は private 追加)

## 設計方針

- deps.json の libdatachannel.tag を v0.24.6 に変更する (url は変更しない)
- CMakeLists.txt は変更しない (v0.24.6 でも同じビルドオプションが使えることを確認する)
- THIRD_PARTY_LICENSES.md は変更しない (libdatachannel と同梱される libjuice / libsrtp / usrsctp のライセンス本文が v0.24.2 と同じであることを確認済み。 バージョンは同ファイルに記載していない)
- CHANGES.md の `## develop` の libdatachannel のエントリは、 2025.1.2 が使っていた v0.24.0 からの最終的な差分を示す 1 件にまとめる (中間状態である v0.24.2 のエントリは v0.24.6 のエントリで置き換える)
- binding していない API (SetThreadPoolSize / Description::sessionId / getNTPOfSR など) への対応は本 issue の範囲外とする
- CHANGES.md の AV1 の未解消経路は v0.24.6 でも残ることを確認済みのため、 記載は変更しない

## 完了条件

- deps.json の libdatachannel.tag が v0.24.6 になっている
- `uv build --wheel` が成功する
- `make develop` の後に `.venv/bin/python -m pytest tests/ -q --timeout=600 --deselect tests/test_peerconnection.py::test_destruct_without_explicit_close --deselect tests/test_peerconnection.py::test_track` が全件 PASS する (`make test` は deselect を渡さず、 `uv run` は sync で `make develop` が入れた拡張モジュールを消す場合があるため使わない)
- prek のフック (ruff / ty / check-toml / check-yaml / clang-format) が PASS する
- CI (wheel.yml の 24 leg と prek.yml) が PASS する
- CHANGES.md の `## develop` に `[UPDATE] libdatachannel のバージョンを v0.24.6 に上げる` と担当者行があり、 中間状態の v0.24.2 のエントリが残っていない
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること

## 参考

- libdatachannel v0.24.3〜v0.24.6: https://github.com/paullouisageneau/libdatachannel/releases
- 対象: deps.json の libdatachannel
- 先行して v0.24.2 へ上げた対応: [[0054-update-libdatachannel-version]]
