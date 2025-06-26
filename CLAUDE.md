# CLAUDE.md

## ビルド

uv run python3 run.py でアーキテクチャを指定する。

```bash
uv run python3 run.py macos_arm64
```

## テスト実行

uv run pytest を利用する。

```bash
uv run pytest test/test_rtp_depacketizer.py
```

## libdatachannel ソース

`_source` ディレクトリに libdatachannel のソースコードが含まれている。
