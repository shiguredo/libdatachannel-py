# OBS Studio WHIP Simulcast 技術仕様書

本ドキュメントは、OBS Studio (commit: a3c2c5df4ad094e08d144bb9f679053a3e9363e6) における WHIP (WebRTC-HTTP Ingestion Protocol) のシミュレキャスト実装について、技術的な詳細を解説したものです。

## 目次

1. [概要](#概要)
2. [アーキテクチャ](#アーキテクチャ)
3. [実装詳細](#実装詳細)
4. [技術仕様](#技術仕様)
5. [実装例](#実装例)

## 概要

OBS Studio の WHIP シミュレキャスト実装は、単一のビデオソースから複数の解像度・ビットレートのストリームを同時に送信する機能を提供します。これにより、受信側は利用可能な帯域幅に応じて最適な品質のストリームを選択できます。

### 主な特徴

- **レイヤー数**: 1〜4レイヤー（UI で設定可能）
- **解像度スケーリング**: 自動的な均等分割
- **ビットレート配分**: 解像度に応じた段階的な配分
- **標準準拠**: WebRTC の simulcast 仕様に準拠

## アーキテクチャ

### 主要コンポーネント

```
┌─────────────────────────────────────────────────────────┐
│                    OBS Studio UI                        │
│                 (Settings → Stream)                     │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                BasicOutputHandler                       │
│            (SimpleOutput/AdvancedOutput)                │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│              WHIPSimulcastEncoders                      │
│         (エンコーダー作成・管理)                           │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                  WHIPOutput                             │
│         (WebRTC 接続・RTP 送信)                          │
└─────────────────────────────────────────────────────────┘
```

## 実装詳細

### 1. 設定と初期化

#### UI 設定

```cpp
// frontend/forms/OBSBasicSettings.ui
<widget class="QSpinBox" name="whipSimulcastTotalLayers">
  <property name="minimum">
    <number>1</number>
  </property>
  <property name="maximum">
    <number>4</number>
  </property>
</widget>
```

#### 設定の読み込み

```cpp
// OBSBasicSettings_Stream.cpp
int whipSimulcastTotalLayers = config_get_int(
    main->Config(), "Stream1", "WHIPSimulcastTotalLayers");
```

### 2. エンコーダー管理 (WHIPSimulcastEncoders)

#### エンコーダー作成アルゴリズム

```cpp
void Create(const char *encoderId, int rescaleFilter, 
            int whipSimulcastTotalLayers, 
            uint32_t outputWidth, uint32_t outputHeight)
{
    // 解像度の計算
    auto widthStep = outputWidth / whipSimulcastTotalLayers;
    auto heightStep = outputHeight / whipSimulcastTotalLayers;

    // レイヤー作成（最高品質を除く）
    for (auto i = whipSimulcastTotalLayers - 1; i > 0; i--) {
        uint32_t width = widthStep * i;
        width -= width % 2;  // 偶数に調整

        uint32_t height = heightStep * i;
        height -= height % 2;  // 偶数に調整

        // エンコーダー作成
        std::string encoder_name = "whip_simulcast_" + std::to_string(i);
        auto encoder = obs_video_encoder_create(
            encoderId, encoder_name.c_str(), nullptr, nullptr);
        
        // スケーリング設定
        obs_encoder_set_scaled_size(encoder, width, height);
        obs_encoder_set_gpu_scale_type(encoder, (obs_scale_type)rescaleFilter);
        
        whipSimulcastEncoders.push_back(encoder);
    }
}
```

#### ビットレート配分

```cpp
void Update(obs_data_t *videoSettings, int videoBitrate)
{
    // 均等分割
    auto bitrateStep = videoBitrate / (whipSimulcastEncoders.size() + 1);
    
    // 低解像度ほど低ビットレート
    for (auto &encoder : whipSimulcastEncoders) {
        videoBitrate -= bitrateStep;
        obs_data_set_int(videoSettings, "bitrate", videoBitrate);
        obs_encoder_update(encoder, videoSettings);
    }
}
```

### 3. RTP ストリーム管理 (WHIPOutput)

#### videoLayerState 構造体

```cpp
struct videoLayerState {
    uint16_t sequenceNumber = 0;     // RTP シーケンス番号
    uint32_t rtpTimestamp = 0;       // RTP タイムスタンプ
    int64_t lastVideoTimestamp = 0;  // 最後のビデオタイムスタンプ
    uint32_t ssrc = 0;               // ストリーム識別子
    std::string rid;                 // Restriction Identifier
};
```

#### RID と SSRC の割り当て

```cpp
// WHIPOutput::Start()
uint32_t base_ssrc = generate_random_u32();

for (uint32_t idx = 0; idx < MAX_OUTPUT_VIDEO_ENCODERS; idx++) {
    auto encoder = obs_output_get_video_encoder2(output, idx);
    if (encoder == nullptr) break;
    
    auto state = std::make_shared<videoLayerState>();
    state->ssrc = base_ssrc + 1 + idx;  // 連続した SSRC
    state->rid = std::to_string(idx);   // "0", "1", "2", "3"
    videoLayerStates[encoder] = state;
}
```

### 4. SDP 生成

#### シミュレキャスト対応 SDP

```cpp
void ConfigureVideoTrack(std::string media_stream_id, std::string cname)
{
    rtc::Description::Video video_description(
        video_mid, rtc::Description::Direction::SendOnly);
    
    // RTP ヘッダー拡張
    video_description.addExtMap(
        rtc::Description::Entry::ExtMap(1, rtpHeaderExtUriMid));  // MID
    video_description.addExtMap(
        rtc::Description::Entry::ExtMap(2, rtpHeaderExtUriRid));  // RID
    
    // 複数レイヤーの場合、RID を追加
    if (videoLayerStates.size() >= 2) {
        // RID を昇順でソート
        std::vector<std::pair<int, std::string>> sortedRids;
        for (const auto &[encoder, state] : videoLayerStates) {
            sortedRids.push_back({std::stoi(state->rid), state->rid});
        }
        std::sort(sortedRids.begin(), sortedRids.end());
        
        // ソート順で RID を追加
        for (const auto &[_, rid] : sortedRids) {
            video_description.addRid(rid);
        }
    }
}
```

生成される SDP の例（3レイヤー）：
```
m=video 9 UDP/TLS/RTP/SAVPF 96
a=mid:video
a=sendonly
a=extmap:1 urn:ietf:params:rtp-hdrext:sdes:mid
a=extmap:2 urn:ietf:params:rtp-hdrext:sdes:rtp-stream-id
a=rid:0 send
a=rid:1 send
a=rid:2 send
a=simulcast:send 0;1;2
```

### 5. RTP パケット送信

```cpp
void Data(struct encoder_packet *packet)
{
    if (video_track && packet->type == OBS_ENCODER_VIDEO) {
        // エンコーダーに対応する状態を取得
        auto state = videoLayerStates[packet->encoder];
        auto rtp_config = video_sr_reporter->rtpConfig;
        
        // RTP 設定を更新
        rtp_config->sequenceNumber = state->sequenceNumber;
        rtp_config->ssrc = state->ssrc;
        rtp_config->rid = state->rid;
        rtp_config->timestamp = state->rtpTimestamp;
        
        // パケット送信
        Send(packet->data, packet->size, duration, 
             video_track, video_sr_reporter);
        
        // 状態を保存
        state->sequenceNumber = rtp_config->sequenceNumber;
        state->rtpTimestamp = rtp_config->timestamp;
    }
}
```

### 6. サーバー応答の検証

```cpp
static size_t simulcast_layers_in_answer(std::string answer)
{
    auto layersStart = answer.find("a=simulcast");
    if (layersStart == std::string::npos) return 0;
    
    auto layersEnd = answer.find("\r\n", layersStart);
    size_t layersAccepted = 1;
    
    // セミコロンの数をカウント
    for (auto i = layersStart; i < layersEnd; i++) {
        if (answer[i] == ';') layersAccepted++;
    }
    
    return layersAccepted;
}
```

## 技術仕様

### 解像度スケーリング

| レイヤー数 | レイヤー 0 | レイヤー 1 | レイヤー 2 | レイヤー 3 |
|-----------|-----------|-----------|-----------|-----------|
| 1         | 100%      | -         | -         | -         |
| 2         | 100%      | 50%       | -         | -         |
| 3         | 100%      | 66%       | 33%       | -         |
| 4         | 100%      | 75%       | 50%       | 25%       |

### ビットレート配分

総ビットレート 3000 kbps、3レイヤーの場合：
- レイヤー 0: 2250 kbps (3000 - 750)
- レイヤー 1: 1500 kbps (2250 - 750)
- レイヤー 2: 750 kbps (1500 - 750)

### RID 命名規則

- 数字文字列: "0", "1", "2", "3"
- 0 が最高品質、数字が大きいほど低品質

### SSRC 割り当て

```
base_ssrc = ランダム値
audio_ssrc = base_ssrc
video_ssrc[0] = base_ssrc + 1  // 最高品質
video_ssrc[1] = base_ssrc + 2
video_ssrc[2] = base_ssrc + 3
video_ssrc[3] = base_ssrc + 4  // 最低品質
```

## 実装例

### libdatachannel-py での実装

```python
class SimulcastLayer:
    def __init__(self, rid: str, width: int, height: int, 
                 bitrate: int, ssrc: int):
        self.rid = rid
        self.width = width
        self.height = height
        self.bitrate = bitrate
        self.ssrc = ssrc
        self.encoder = None
        self.rtp_config = None

class WHIPSimulcastClient:
    def _create_simulcast_layers(self):
        base_ssrc = random.randint(0, 0xFFFFFFFF)
        layer_configs = [
            {"rid": "3", "scale": 1.0, "bitrate_ratio": 1.0},
            {"rid": "2", "scale": 0.75, "bitrate_ratio": 0.6},
            {"rid": "1", "scale": 0.5, "bitrate_ratio": 0.3},
            {"rid": "0", "scale": 0.25, "bitrate_ratio": 0.1},
        ]
        
        start_idx = len(layer_configs) - self.simulcast_layers
        for i, config in enumerate(layer_configs[start_idx:]):
            width = int(self.video_width * config["scale"])
            height = int(self.video_height * config["scale"])
            bitrate = int(self.base_bitrate * config["bitrate_ratio"])
            
            layer = SimulcastLayer(
                rid=config["rid"],
                width=width,
                height=height,
                bitrate=bitrate,
                ssrc=base_ssrc + 1 + i
            )
            self.layers.append(layer)
```

### SDP 操作

```python
def _add_simulcast_to_sdp(self, sdp: str) -> str:
    lines = sdp.split('\n')
    modified_lines = []
    
    for line in lines:
        modified_lines.append(line)
        
        if line.startswith('m=video'):
            # RID と simulcast 属性を追加
            rid_list = ";".join([layer.rid for layer in self.layers])
            modified_lines.append(f"a=simulcast:send {rid_list}")
            
            for layer in self.layers:
                modified_lines.append(f"a=rid:{layer.rid} send")
    
    return '\n'.join(modified_lines)
```

## まとめ

OBS Studio の WHIP シミュレキャスト実装は、WebRTC の標準仕様に準拠しながら、実用的で効率的な実装となっています。主な特徴：

1. **シンプルな解像度スケーリング**: 均等分割による予測可能な品質階層
2. **効率的なビットレート配分**: 解像度に応じた段階的な配分
3. **標準準拠の SDP**: WebRTC シミュレキャスト仕様に完全準拠
4. **堅牢なエラーハンドリング**: サーバー側の制限に適切に対応
5. **拡張性**: 最大4レイヤーまでサポート、将来的な拡張も容易

この実装により、ネットワーク状況に応じた適応的なビデオストリーミングが可能となり、視聴者により良い体験を提供できます。