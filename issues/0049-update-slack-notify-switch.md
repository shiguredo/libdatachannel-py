# Slack 通知をリポジトリ変数で一時的に止められるようにする

- Priority: High
- Created: 2026-10-10
- Completed: 2026-10-10
- Branch: feature/update-slack-notify-switch
- Polished: 2026-10-10

## 目的

Slack 通知が不要な期間に、 コードを書き換えずに止められるようにする。 現在は `wheel.yml` の push ごとの実行で通知が飛ぶため、 集中して作業している間は止めたいという要望がある。

## 優先度根拠

- 通知を止める手段がワークフローの編集 (PR + CI + マージ) しか無く、 止めるまでに数分かかる
- リポジトリ変数で切り替えられれば、 再開もコマンド 1 つで済む

## 現状

- `.github/workflows/wheel.yml` に通知が 2 箇所ある (`slack_notify` ジョブと、 タグ push 時のリリース失敗通知)
- どちらも `slack_channel: python-oss` へ送る (issue 0048 で変更済み)

## 設計方針

- リポジトリ変数 `SLACK_NOTIFY` が `false` のときは通知ジョブを実行しない (`vars.SLACK_NOTIFY != 'false'` を `if:` に足す)
- 変数が未設定の場合と `true` の場合は従来どおり通知する (既定で止まらないようにする)

## 完了条件

- `.github/workflows/wheel.yml` の通知 2 箇所が `vars.SLACK_NOTIFY != 'false'` で保護されていること
- `prek run --all-files check-yaml` が PASS すること
- リポジトリ変数 `SLACK_NOTIFY` に `false` を設定して通知が止まること (設定手順を解決方法に残す)
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること

## 解決方法

- `.github/workflows/wheel.yml` の通知 2 箇所の `if:` に `vars.SLACK_NOTIFY != 'false'` を追加した (変数が未設定または `true` なら従来どおり通知する)
- リポジトリ変数 `SLACK_NOTIFY=false` を設定し、 通知を止めた。 再開は `gh variable set SLACK_NOTIFY --body true`、 または変数を削除する
- 検証
  - `prek run --all-files check-yaml` が PASS
  - `gh variable list` で `SLACK_NOTIFY=false` が設定されていることを確認した
  - `/review-diff-code` の致命的 / 重要指摘が 0 件
