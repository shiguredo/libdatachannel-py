# media() / application() / rtp_map() / config() が親の寿命に紐付かない参照を返す

- Priority: High
- Created: 2026-08-30
- Completed: {YYYY-MM-DD}
- Branch: feature/fix-non-owning-reference-lifetime
- Polished: 2026-10-09

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

- `get_media` は `nb::cast(Description::Media* / Application*)` で pointer を返す。 nanobind の既定 policy は pointer に対して reference (non-owning) に確定する
- `application` / `rtp_map` / `config` は `nb::rv_policy::reference` が明示されているが、 keep_alive は未指定
- いずれも libdatachannel 内部 (`Description::mEntries` / `Description::Media::mRtpMaps` / impl が保持する Configuration) への参照であり、 親の破棄とともに無効になる
- `Description::Media::rtpMap(int)` は `RtpMap*` を返し、 `removeRtpMap` で erase されると無効になる (値のコピーを返せば防げる)
- `PeerConnection::config()` は `const Configuration*` を返す
- `rtp_map` は存在しない payload type に対しては `ValueError` になる (libdatachannel 側が例外を投げる。 この挙動は変更しない)
- keep_alive の全数調査では、 この 4 箇所以外の参照返却は存在せず、 他は値または shared_ptr で問題なし

## 設計方針

- media / application / config は `nb::cast(..., nb::rv_policy::reference_internal, nb::find(parent))` で親を渡し、 戻り値が親を生存させる形にする (`nb::object` を返す関数では def 側の policy が効かないため、 cast に parent を渡す)
- rtp_map は `remove_rtp_map` による erase 経路があるため keep_alive では防げない。 値 (コピー) を `std::optional` で返す設計に変更する (存在しない payload type は従来どおり例外)
- 動作変更に合わせて型スタブを再生成 (`make develop`) し、 テストを更新する
- 寿命の検証は回帰時にプロセスが落ちるため、 テストの実行中に SEGV したらそのまま失敗として扱う (サブプロセス分離はしない。 落ちれば即座に CI が失敗するため)

## 完了条件

- 親を先に破棄した後に子を触っても crash しないこと
- rtp_map が `remove_rtp_map` 後も有効な値を返すこと
- 実測した 3 経路 (media / application / rtp_map) の回帰テストが追加されていること
- `prek run --all-files pytest` (prek.toml の pytest フック = 既知の恒停テストを `--deselect` で除外) が PASS する
- CI (wheel.yml の leg / prek.yml の `ty` ジョブ) が PASS する
- `CHANGES.md` の `## develop` に `[FIX]` エントリが追加されている
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること

## 参考

- 対象シンボル: `get_media`、`bind_description`、`bind_peerconnection` (src/bind_libdatachannel.cpp)
- libdatachannel v0.24.0: `include/rtc/description.hpp` (`Description::media(int)`、`Description::application()`、`Description::Media::rtpMap(int)`)
