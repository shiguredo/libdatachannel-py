# libdatachannel-py Examples

このディレクトリには libdatachannel-py の使用例が含まれています。

## セットアップ

まず、ルートディレクトリで libdatachannel-py をビルドしてください：

```bash
cd ..
uv run python run.py macos_arm64  # macOS ARM64 の場合
```

## 実行方法

examples は UV のワークスペース機能を使用しているため、`uv run` コマンドで実行します：

```bash
# examples ディレクトリから実行
uv run python whip.py --url https://example.com/whip --duration 30

# ルートディレクトリから実行
uv run --package examples python examples/whip.py --url https://example.com/whip
```

## サンプル一覧

### whip.py

WHIP (WebRTC-HTTP Ingestion Protocol) クライアントの実装例です。

- **プロトコル**: RFC 9725 準拠の WHIP クライアント
- **ビデオコーデック**: AV1
- **オーディオコーデック**: Opus
- **機能**:
  - HTTP POST でオファーを送信
  - 201 Created レスポンスでアンサーを受信
  - Location ヘッダーを使用したセッション管理
  - カメラ/マイクからのリアルタイム配信
  - テストパターン（カラーバーとサイン波）の生成

使用例：

```bash
# テストパターンで10秒間送信
uv run python whip.py

# カメラとマイクを使用して配信
uv run python whip.py --camera --microphone --duration 30

# カメラのみ使用（音声はテストトーン）
uv run python whip.py --camera --url https://whip.example.com/live

# マイクのみ使用（映像はテストパターン）
uv run python whip.py --microphone --duration 60

# デバッグログを有効化
uv run python whip.py --camera --microphone --debug
```

オプション：

- `--url`: WHIP エンドポイントの URL（デフォルト: <https://example.com/whip）>
- `--duration`: 配信時間（秒）（指定しない場合は無期限に配信）
- `--camera`: カメラを使用（デフォルト: テストパターン）
- `--microphone`: マイクを使用（デフォルト: テストトーン）
- `--debug`: デバッグログを有効化

無期限配信の例：

```bash
# テストパターンを無期限に配信（Ctrl+C で停止）
uv run python whip.py --url http://127.0.0.1:5000/whip/sora

# カメラとマイクで無期限配信
uv run python whip.py --camera --microphone --url http://127.0.0.1:5000/whip/sora
```

## 依存関係

examples プロジェクトは以下の依存関係を持っています：

- libdatachannel-py（ローカルビルド）
- httpx（HTTP クライアント）
- numpy（数値計算・信号生成）
- opencv-python（カメラキャプチャ）
- sounddevice（音声キャプチャ）

## 開発

新しいサンプルを追加する場合：

1. このディレクトリに新しい Python ファイルを作成
2. 必要に応じて `pyproject.toml` の dependencies に依存関係を追加
3. この README.md にサンプルの説明を追加

## トラブルシューティング

### ImportError が発生する場合

libdatachannel-py が正しくビルドされていることを確認してください：

```bash
cd ..
uv run python run.py macos_arm64
```

### WHIP サーバーへの接続エラー

- URL が正しいことを確認
- ネットワーク接続を確認
- WHIP サーバーが稼働していることを確認
- `--debug` オプションで詳細なログを確認

### カメラ/マイクが認識されない場合

- カメラへのアクセス権限を確認
- マイクへのアクセス権限を確認
- 他のアプリケーションがデバイスを使用していないか確認
