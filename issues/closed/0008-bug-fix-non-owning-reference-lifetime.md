# media() / application() / rtp_map() / config() が親の寿命に紐付かない参照を返す

- Priority: High
- Created: 2026-08-30
- Completed: 2026-10-09
- Branch: feature/fix-non-owning-reference-lifetime
- Polished: 2026-10-09

## 目的

Description.media() / Description.application() / Media.rtp_map() / PeerConnection.config() が C++ 内部オブジェクトへの non-owning 参照を返すにもかかわらず keep_alive を持たないため、親オブジェクトを先に破棄するだけで use-after-free になる。自然な Python コードで SEGV する寿命問題を解消する。

## 優先度根拠

- 実測 (修正前のビルド): `m = desc.media(0)` のように親を破棄した後に `m.mid()` を呼ぶと exit 139 (SIGSEGV) になる。 同様の経路 (`application()` / `config()`) も親を破棄すると落ちる
- 実測 (修正前のビルド): `rtp_map()` の戻り値は `remove_rtp_map()` 後に値が破壊される (`format` が空文字列になる)
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
- rtp_map は `remove_rtp_map` による erase 経路があるため keep_alive では防げない。 値 (コピー) を返す設計に変更する (存在しない payload type は libdatachannel が例外を投げ、 従来どおり `ValueError` になる)
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

## 解決方法

- `src/bind_libdatachannel.cpp` で戻り値が親を生存させるようにした
  - `media()`: `nb::cast(ptr, nb::rv_policy::reference_internal, nb::find(desc))` で親 (Description) を保持する
  - `application()` / `config()`: メンバ関数ポインタ + `nb::rv_policy::reference_internal` (self が parent になる) に変更した。 `application()` は const / 非 const の overload があるため `nb::overload_cast<>` で非 const 版を選ぶ
  - 生成スタブも `application -> Description.Application` / `config -> Configuration` と正確になった
- `rtp_map()` は値 (コピー) を返すようにした (`remove_rtp_map()` / `remove_format()` の erase で無効になるため)。 存在しない payload type は従来どおり `ValueError` になる
- テスト
  - `tests/test_description.py`: 親を `del` + `gc.collect()` で破棄した後に `media.mid()` / `application.mid()` を読むこと、 `rtp_map()` が値 (コピー) を返すこと (書き換えが反映されない / 呼ぶたびに別オブジェクト) を検証する
  - `tests/test_peerconnection.py`: 親を破棄した後に `config.ice_servers` を読むことを検証する
- `CHANGES.md` の `## develop` に `[CHANGE]` (rtp_map の値返し) と `[FIX]` (media / application / config の寿命) を追加した
- 実測: 修正前は親を破棄した後の `mid()` で exit 139 (SIGSEGV)、 `rtp_map()` は `remove_rtp_map()` 後に `format` が空文字列。 修正後は media / application / config / rtp_map の 4 経路すべて正常で、 全体 94 passed / 12 skipped / 1 deselected

## スコープ外 (関連する未解決問題)

- **`clear_media()` を呼ぶと、 それ以前に取得した `media()` / `application()` の戻り値は無効になる** (`Description::clearMedia()` が `mEntries.clear()` / `mApplication.reset()` で実体を解放するため)。 親を生存させるだけでは防げず、 binding 側で対処するには `media()` を値返しにする (内部を書き換える既存の使い方を壊す) などの設計変更が要る。 実測では修正後も `desc.media(0)` → `desc.clear_media()` → `m.mid()` で exit 139 (SIGSEGV) になる。 恒久対応は [[0043-bug-fix-clear-media-invalidates-references]] で扱う
- `remove_format` / `remove_rtp_map` による `rtp_map()` の無効化は、 値 (コピー) を返すようにしたことで解消済み

## 参考

- 対象シンボル: `get_media`、`bind_description`、`bind_peerconnection` (src/bind_libdatachannel.cpp)
- libdatachannel v0.24.0: `include/rtc/description.hpp` (`Description::media(int)`、`Description::application()`、`Description::Media::rtpMap(int)`)
