# MediaHandler改善提案

## 現在の問題

1. **MediaHandlerチェーンの出力が取得できない**
   - `RtpDepacketizer`や`H264RtpDepacketizer`をセットしても、処理結果がPythonで受け取れない
   - `Track.on_message()`には常に生のRTPパケットが渡される

2. **回避策の必要性**
   - 現在は手動でRTPヘッダーを解析する必要がある
   - MediaHandlerの本来の機能が使えていない

## 提案する解決策

### 1. MediaHandlerのPythonラッパー改善

```cpp
// bind_libdatachannel.cppに追加
class PyMediaHandlerWrapper : public MediaHandler {
private:
    std::function<void(binary)> m_callback;
    std::shared_ptr<MediaHandler> m_wrapped;
    
public:
    PyMediaHandlerWrapper(std::shared_ptr<MediaHandler> wrapped, 
                         std::function<void(binary)> callback)
        : m_wrapped(wrapped), m_callback(callback) {}
    
    void incoming(message_vector &messages, const message_callback &send) override {
        // 元のハンドラーで処理
        m_wrapped->incoming(messages, send);
        
        // 処理結果をPythonコールバックに渡す
        for (auto &msg : messages) {
            if (msg->type == Message::Binary) {
                m_callback(*msg);
            }
        }
    }
};
```

### 2. Track APIの拡張

```python
# 新しいAPI案
track.set_media_handler(rtcp_session)
track.chain_media_handler(rtp_depacketizer)

# MediaHandler処理後のデータを受け取る新しいコールバック
track.on_depacketized_message(callback)
```

### 3. 即座に使える回避策

現在のバインディングでも動作する`custom_media_handler.py`を提供：
- `RtpDepacketizingHandler`: RTPヘッダーを手動で除去
- `CallbackMediaHandler`: 処理結果をPythonコールバックに渡す

## 使用例

```python
# カスタムハンドラーを使用
from custom_media_handler import RtpDepacketizingHandler

def on_video_payload(payload, pt, timestamp):
    # ペイロードを処理
    print(f"Received payload: PT={pt}, size={len(payload)}")

# ハンドラーをセット
depacketizer = RtpDepacketizingHandler(on_video_payload)
track.chain_media_handler(depacketizer)
```

## 今後の方向性

1. **短期的**: カスタムMediaHandlerで回避
2. **中期的**: Pythonバインディングの改善
3. **長期的**: libdatachannel本体でPython向けのAPIを追加