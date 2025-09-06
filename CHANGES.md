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

### misc

- [CHANGE] リリースを GH コマンドに変更
  - `gh release create` を使用してリリースを作成するように変更する
  - @voluntas
- [CHANGE] run.py に build と format サブコマンドを追加
  - .github/workflows/build.yml も修正
  - @voluntas
- [ADD] デバッグビルドの追加
  - ローカルバージョンラベル +debug を指定している
  - @voluntas
