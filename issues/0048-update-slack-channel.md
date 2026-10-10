# Slack 通知のチャンネルを python-oss にする

- Priority: Medium
- Created: 2026-10-10
- Completed: {YYYY-MM-DD}
- Branch: feature/update-slack-channel
- Polished: 2026-10-10

## 目的

CI・リリースの結果を通知する Slack のチャンネルが `sora-python-sdk` のままになっている。 本リポジトリ (libdatachannel-py) の通知先である `python-oss` に直す。

## 優先度根拠

- `.github/workflows/wheel.yml` の `slack_notify` ジョブが `slack_channel: sora-python-sdk` を指定しており、 別プロダクトのチャンネルへ通知されている
- 本リポジトリは Sora Python SDK ではなく libdatachannel の Python バインディングであり、 通知先が誤っていると障害に気づくべき人が気づけない

## 現状

- `.github/workflows/wheel.yml` の `slack_notify` ジョブ (build_ubuntu / build_macos の完了後に実行) が `slack_channel: sora-python-sdk` を渡している
- 他リポジトリでは `rust-oss` などのプロダクト別チャンネルを使っている (`shiguredo/github-actions` の `slack-notify`)

## 設計方針

- `slack_channel` を `python-oss` に変更する。 ジョブの構成 (`needs` / `if` / `permissions` / `notify_mode`) は他リポジトリと揃っているため変更しない

## 完了条件

- `.github/workflows/wheel.yml` の `slack_notify` ジョブの `slack_channel` が `python-oss` であること
- `prek run --all-files check-yaml` が PASS すること
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること

## 参考

- 実装例: `audio-toolbox-rs` の `.github/workflows/release.yml` の `slack_notify` ジョブ
- 通知ジョブは `wheel.yml` のみにあり、 `prek.yml` (pull_request / push の CI) には無い
