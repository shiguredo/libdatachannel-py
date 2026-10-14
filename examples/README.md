# libdatachannel-py Examples

このディレクトリには libdatachannel-py の使用例が含まれています。

## セットアップ

リポジトリのルートディレクトリで、 サンプルの依存ライブラリと拡張モジュールを用意します。

```bash
# サンプルの依存ライブラリ (uvc-py / portaudio-py / blend2d-py / webcodecs-py / raw-player など)
make example

# 拡張モジュールをビルドして .venv に install する
make develop
```

## 実行方法

ルートディレクトリから `uv run python` で実行します。

```bash
uv run python examples/whip.py --url https://example.com/whip --fake-capture-device --duration 10
```

`libdatachannel` の import に失敗する場合は `make develop` を実行し直してください。

## サンプル一覧

### whip.py

WHIP (WebRTC-HTTP Ingestion Protocol, RFC 9725) クライアントの実装例です。
webcodecs-py でエンコードし、 libdatachannel-py で配信します。

- 映像コーデックは `--video-codec-type` で h264 / h265 / av1 から選ぶ (既定は h264)
- `--fake-capture-device` を付けると blend2d-py のテストパターンとテスト音声を使う
- 付けない場合は uvc-py のカメラと portaudio-py のマイクを使う

```bash
# テストパターンとテスト音声で 10 秒配信する
uv run python examples/whip.py --url https://example.com/whip --fake-capture-device --duration 10

# カメラとマイクで配信する
uv run python examples/whip.py --url https://example.com/whip

# デバイス番号とビットレートを指定する
uv run python examples/whip.py --url https://example.com/whip --video-input-device 1 --audio-input-device 1 --bitrate 3000000
```

主なオプション:

- `--url`: WHIP エンドポイントの URL (必須)
- `--token`: Bearer トークン (認証用)
- `--duration`: 配信時間 (秒)。 指定しない場合は Ctrl+C まで配信する
- `--video-codec-type`: 映像コーデック (h264 / h265 / av1。 既定は h264)
- `--fake-capture-device`: テストパターンとテスト音声を使う
- `--video-input-device` / `--audio-input-device`: 入力デバイス番号
- `--framerate` / `--bitrate`: フレームレート (既定 30) とビットレート (既定 5000000)
- `--disable-audio-processing`: 音声のエンコードと送信を無効にする (デバッグ用)
- `--debug`: デバッグログを出力する

### whep.py

WHEP (WebRTC-HTTP Egress Protocol) クライアントの実装例です。
libdatachannel-py で受信し、 webcodecs-py でデコードします。

- `--video-codec-type` (h264 / h265) は必須
- `--display` を付けると raw-player で映像と音声を再生する

```bash
# 10 秒受信する
uv run python examples/whep.py --url https://example.com/whep --video-codec-type h264 --duration 10

# 映像と音声を再生する
uv run python examples/whep.py --url https://example.com/whep --video-codec-type h264 --display
```

主なオプション:

- `--url`: WHEP エンドポイントの URL (必須)
- `--video-codec-type`: 映像コーデック (h264 / h265。 必須)
- `--token`: Bearer トークン (認証用)
- `--duration`: 受信時間 (秒)。 指定しない場合は Ctrl+C まで受信する
- `--display`: 映像と音声を表示・再生する
- `--debug`: デバッグログを出力する

### 共有モジュール

`whip.py` と `whep.py` は次のモジュールを共有しています。 単体で実行するものではありません。

- `error_logging.py`: structlog を使ったエラーログの共通処理
- `rtp_timestamp.py`: dts (マイクロ秒) から RTP timestamp を計算する
- `trickle_ice.py`: Trickle ICE の SDP fragment を組み立てる

## 依存関係

- libdatachannel-py (このリポジトリのローカルビルド)
- blend2d-py (テストパターンの描画)
- httpx (HTTP クライアント)
- portaudio-py (音声入力)
- raw-player (映像と音声の再生)
- structlog (ログ)
- uvc-py (カメラ入力)
- webcodecs-py (エンコード / デコード)

## 開発

新しいサンプルを追加する場合は、 このディレクトリにファイルを追加し、 ルートの `pyproject.toml` の `example` グループに依存を追加してから、 この README に説明を追加してください。

## トラブルシューティング

### `libdatachannel` の import に失敗する場合

`make develop` を実行し、 拡張モジュールをビルドして `.venv` に install してください。

### WHIP / WHEP サーバーへの接続エラー

- `--url` が正しいことを確認する
- ネットワーク接続を確認する
- WHIP / WHEP サーバーが稼働していることを確認する
- `--debug` を付けて詳細なログを確認する

### カメラ / マイクが認識されない場合

- カメラとマイクへのアクセス権限を確認する
- 他のアプリケーションがデバイスを使用していないか確認する
- `--video-input-device` / `--audio-input-device` でデバイス番号を指定する
