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

- [UPDATE] build_pyi ジョブを matrix 化し Python バージョンごとに pyi を生成
  - Python 3.13 と 3.14 の両方で pyi を生成するように変更
  - 各ビルドジョブで対応する Python バージョンの pyi をダウンロード
  - @voluntas

### misc

- [CHANGE] リリースを GH コマンドに変更
  - `gh release create` を使用してリリースを作成するように変更する
  - @voluntas
- [CHANGE] run.py に build と format サブコマンドを追加
  - .github/workflows/build.yml も修正
  - @voluntas
- [UPDATE] actions/checkout と actions/download-artifact を v5 に上げる
  - @miosakuma
- [ADD] デバッグビルドの追加
  - ローカルバージョンラベル +debug を指定している
  - @voluntas
