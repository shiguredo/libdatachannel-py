# 変更履歴

- CHANGE
  - 後方互換性のない変更
- UPDATE
  - 後方互換性がある変更
- ADD
  - 後方互換性がある追加
- FIX
  - バグ修正

## develop

- [CHANGE] RtpDepacketizer、H264RtpDepacketizer、MediaHandler の incoming() メソッドの動作を変更
  - incoming() メソッドが処理後のメッセージリストを返すように変更
  - これにより、ペイロードの抽出と無効なパケットの破棄が正しく動作するようになった
  - @voluntas

### misc

- [CHANGE] リリースを GH コマンドに変更
  - `gh release create` を使用してリリースを作成するように変更する
  - @voluntas
- [ADD] デバッグビルドの追加
  - ローカルバージョンラベル +debug を指定している
  - @voluntas
