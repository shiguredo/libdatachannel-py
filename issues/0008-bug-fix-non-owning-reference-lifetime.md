# media() / application() / rtp_map() / config() が親の寿命に紐付かない参照を返す

- Priority: High
- Created: 2026-08-30
- Completed: {YYYY-MM-DD}
- Branch: feature/fix-non-owning-reference-lifetime
- Polished: {YYYY-MM-DD}

## 目的

Description.media() / Description.application() / Media.rtp_map() / PeerConnection.config() が C++ 内部オブジェクトへの non-owning 参照を返すにもかかわらず keep_alive を持たないため、親オブジェクトを先に破棄するだけで use-after-free になる。自然な Python コードで SEGV する寿命問題を解消する。

## 優先度根拠

- 実測: `m = desc.media(0)` → `del desc` → `gc.collect()` → `m.mid()` で SIGSEGV。application も同様
- 実測: `rtp_map()` の戻り値は `remove_rtp_map()` 後に値が破壊される (format が空文字列になる)
- Python の参照カウントは即時解放のため、「関数内で Description を作って media だけ返す」コードで確実に発火する
- Free-Threading 環境では GC タイミングが非決定的になり、顕在化しやすい

## 現状

再現手順:

```python
from libdatachannel import Description

def make():
    desc = Description("v=0...")
    desc.add_audio("audio", Description.Direction.SendOnly)
    return desc.media(0)

m = make()
m.mid()  # 親の Description は破棄済み → use-after-free
```

- `get_media` は `nb::cast(Description::Media* / Application*)` で pointer を返す。nanobind の既定 policy は pointer に対して reference (non-owning) に確定する
- `application` / `rtp_map` / `config` は `nb::rv_policy::reference` が明示されているが、keep_alive は未指定
- いずれも libdatachannel 内部 (`Description::mEntries` / `Description::Media::mRtpMaps` / impl が保持する Configuration) への参照であり、親の破棄とともに無効になる
- keep_alive の全数調査では、この 4 箇所以外の参照返却は存在せず、他は値または shared_ptr で問題なし

## 設計方針

- media / application / config は `nb::rv_policy::reference_internal` と keep_alive を付与し、戻り値が親を保持する形にする
- rtp_map は `remove_rtp_map` による erase 経路があるため keep_alive では防げない。値 (コピー) を返す設計に変更する
- 動作変更に合わせて型スタブを再生成し、テストを更新する

## 完了条件

- 親を先に破棄した後に子を触っても crash しないこと
- rtp_map が `remove_rtp_map` 後も有効な値を返すこと
- 実測した 3 経路 (media / application / rtp_map) の回帰テストが追加されていること
- `uv sync && make test` で全テストが PASS すること
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること

## 参考

- 対象シンボル: `get_media`、`bind_description`、`bind_peerconnection` (src/bind_libdatachannel.cpp)
- libdatachannel v0.24.0: `include/rtc/description.hpp` (`Description::media(int)`、`Description::application()`、`Description::Media::rtpMap(int)`)
