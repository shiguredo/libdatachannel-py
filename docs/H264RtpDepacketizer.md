# H264RtpDepacketizer クラスの詳細解説

## 概要

`H264RtpDepacketizer` は、RTPパケットからH.264ビデオデータを抽出し、再構築するためのクラスです。このクラスは `MediaHandler` を継承しており、WebRTCのメディアパイプラインの一部として動作します。

## 主な機能

### 1. RTPパケットのデパケタイゼーション

RTPで送信されたH.264ビデオデータは、ネットワーク伝送のために分割されています。このクラスは以下の形式をサポートしています：

- **Single NAL Unit Packets**: 1つのRTPパケットに1つのNALユニット
- **FU-A (Fragmentation Unit Type A)**: 大きなNALユニットを複数のRTPパケットに分割
- **STAP-A (Single Time Aggregation Packet Type A)**: 複数の小さなNALユニットを1つのRTPパケットに集約

### 2. NALユニットの再構築

分割されたNALユニットを元の形に再構築し、デコーダーで処理可能な形式に変換します。

## クラス構造

```cpp
class H264RtpDepacketizer : public MediaHandler {
public:
    using Separator = NalUnit::Separator;
    
    // コンストラクタ - セパレータタイプを指定
    H264RtpDepacketizer(Separator separator = Separator::LongStartSequence);
    
    // RTPパケットを処理するメインメソッド
    void incoming(message_vector &messages, const message_callback &send) override;

private:
    std::vector<message_ptr> mRtpBuffer;  // RTPパケットのバッファ
    const NalUnit::Separator mSeparator;  // NALユニットセパレータ
    
    void addSeparator(binary &accessUnit);
    message_vector buildFrames(message_vector::iterator firstPkt, 
                              message_vector::iterator lastPkt,
                              uint8_t payloadType, uint32_t timestamp);
};
```

## 動作の詳細

### 1. RTPパケットの受信と蓄積

`incoming()` メソッドが呼ばれると：

1. 受信したメッセージからRTPパケットをフィルタリング
2. RTPパケットを内部バッファ（`mRtpBuffer`）に蓄積
3. 同じタイムスタンプのパケットをグループ化

### 2. NALユニットの処理

`buildFrames()` メソッドで、同じタイムスタンプのRTPパケットから完全なフレームを構築：

#### FU-A (Fragmentation Unit) の処理
- NALユニットタイプが28の場合
- Start bitがセットされている場合は、新しいNALユニットの開始
- フラグメントヘッダーから元のNALユニットタイプを復元
- 各フラグメントのペイロードを結合

#### Single NAL Unit の処理
- NALユニットタイプが1-23の場合
- RTPペイロード全体が1つのNALユニット
- セパレータを追加してアクセスユニットに追加

#### STAP-A (Aggregation Packet) の処理
- NALユニットタイプが24の場合
- 1つのRTPパケットに複数のNALユニットが含まれる
- 各NALユニットのサイズを読み取り、個別に抽出

### 3. セパレータの追加

デコーダーがNALユニットの境界を識別できるよう、各NALユニットの前にセパレータを追加：

- **LongStartSequence**: `0x00 0x00 0x00 0x01`
- **ShortStartSequence**: `0x00 0x00 0x01`
- **Length**: 最初の4バイトにNALユニットの長さを格納

## 使用方法

```cpp
// H264RtpDepacketizerのインスタンスを作成
auto depacketizer = std::make_shared<H264RtpDepacketizer>(
    NalUnit::Separator::LongStartSequence
);

// メディアハンドラーチェーンに追加
track->addMediaHandler(depacketizer);
```

## 重要な定数

- `naluTypeSTAPA` (24): STAP-Aパケットタイプ
- `naluTypeFUA` (28): FU-Aフラグメンテーションユニットタイプ

## エラーハンドリング

- 不正なセパレータタイプが指定された場合は例外をスロー
- STAP-Aパケットで宣言されたサイズがバッファサイズを超える場合は例外をスロー
- 未知のRTPパケタイゼーション形式の場合は例外をスロー

## 出力形式

処理されたNALユニットは、`Message` オブジェクトとして出力されます：
- `Message::Binary` タイプ
- `FrameInfo` にペイロードタイプとタイムスタンプを含む
- NALユニットの前に指定されたセパレータが付加される

## 注意事項

1. このクラスは同じタイムスタンプのRTPパケットが全て到着するまで待機します
2. パケットの順序は保証されていないため、内部でバッファリングして再構築します
3. `RTC_ENABLE_MEDIA` マクロが定義されている場合のみ使用可能です

## 関連クラス

- `MediaHandler`: 基底クラス
- `NalUnit`: NALユニットの構造を定義
- `RtpHeader`: RTPヘッダーの構造を定義
- `FrameInfo`: フレーム情報（ペイロードタイプ、タイムスタンプ）を保持