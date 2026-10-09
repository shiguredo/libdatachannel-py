# clear_media() が media() / application() の取得済み参照を無効にする

- Priority: High
- Created: 2026-10-09
- Completed: {YYYY-MM-DD}
- Branch: feature/fix-clear-media-invalidates-references
- Polished: {YYYY-MM-DD}

## 目的

`Description` には、 それ以前に取得した `media()` / `application()` の戻り値を無効にする操作がある (`clear_media()` は `mEntries.clear()` / `mApplication.reset()`、 `add_media(Application)` / `add_application()` は `removeApplication()`)。 戻り値は内部オブジェクトへの生ポインタのため、 無効化後に触ると use-after-free でプロセスが落ちるか、 別オブジェクトを指す。 親を生存させるだけでは防げないため、 安全な扱いを確定する。

## 優先度根拠

- 実測 ([[0008-bug-fix-non-owning-reference-lifetime]] の修正後ビルド): `desc.add_audio("audio", ...)` → `m = desc.media(0)` → `desc.clear_media()` → `m.mid()` で exit 139 (SIGSEGV)
- [[0008-bug-fix-non-owning-reference-lifetime]] で親の寿命に紐付ける修正を入れたが、 この経路は解放の主体が親ではなく破壊的操作のため解消していない
- `clear_media()` は SDP を組み直す際に使われ得る操作であり、 取得済みの `Media` を保持したまま呼ぶコードは自然に書ける

## 現状

再現手順 (修正後のビルドでも再現する):

```python
from libdatachannel import Description

desc = Description("v=0...")
desc.add_audio("audio", Description.Direction.SendOnly)
m = desc.media(0)
desc.clear_media()
m.mid()  # 解放済みの Media を参照 → SIGSEGV (exit 139)
```

- `_deps/libdatachannel/v0.24.0/source/src/description.cpp` の `clearMedia()` は `mEntries.clear(); mApplication.reset();`
- `Description::mEntries` は `std::vector<std::shared_ptr<Entry>>` で、 Python 側が保持しているのは `Entry` 内の `Media*` / `Application*` への生ポインタ
- binding 側で `shared_ptr` を取得する公開 API は無く、 `media(int)` / `application()` は生ポインタを返す
- `add_media(Application)` / `add_application()` も `removeApplication()` を先に呼ぶため、 取得済みの `application()` 参照を無効にする (実測: 旧参照が新しい Application を指し、 `sctp_port()` が None になる)
- `add_media(Media)` / `add_video()` / `add_audio()` / `add_rtp_map()` は `mEntries` (`vector<shared_ptr<Entry>>`) に追加するだけで、 取得済みの `Media*` を無効にしない

## 設計方針

- 対処の候補を実測して決める:
  - 案 A: `media()` / `application()` を値 (コピー) で返す。 安全だが、 戻り値を書き換えると `Description` に反映される既存の使い方 (`desc.media(0).set_bitrate(64000)` / `desc.application().set_sctp_port(5000)` 等) が壊れる
  - 案 B: `Description` 側で解放を遅延させる (取得済みの `Entry` への `shared_ptr` を Python 側が保持する) 方法。 公開 API に `shared_ptr` を返す経路が無いため、 libdatachannel 側の変更か、 binding 側で `Entry` を複製保持する仕組みが要る
  - 案 C: 仕様として明記する (`clear_media()` は取得済み参照を無効にする)。 変更が小さく、 C++ の `std::vector` の `clear()` と同じ意味論になる
- どの案を採るかは、 既存の example / テスト / whip 実装での `clear_media()` の使われ方と、 API 互換性への影響を実測してから決める
- 案 C を採る場合は、 docstring と CHANGES に明記し、 誤用時に落ちることを防ぐ案内を書く

## 完了条件

- 取得済みの `media()` / `application()` 参照を無効にする操作を一覧化する (`clear_media()` / `add_media(Application)` / `add_application()`)
- 採用した案ごとに検証方法を定めて確認する: 案 A は `clear_media()` 後の取得済み参照が有効なコピーであること、 案 B は取得済み参照が `clear_media()` 後も同じオブジェクトを指すこと、 案 C は無効化される操作と無効化後の扱いが docstring と CHANGES に明記されていること
- 案 A / B を採る場合は、 既存の `media()` 経由の書き換えが引き続き動くことをテストで確認する
- `prek run --all-files pytest` が PASS する
- CI (wheel.yml の leg / prek.yml の `ty` ジョブ) が PASS する
- `CHANGES.md` の `## develop` に変更内容が記録されている
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること

## スコープ外 (関連する未解決問題)

- `remove_format` / `remove_rtp_map` による `rtp_map()` の無効化は [[0008-bug-fix-non-owning-reference-lifetime]] の値返しで解消済み
- `as_audio()` / `as_video()` の値コピーは [[0013-bug-fix-as-audio-as-video]] で扱う

## 参考

- libdatachannel v0.24.0: `source/src/description.cpp` の `clearMedia()` / `removeRtpMap()` / `removeFormat()`、 `source/include/rtc/description.hpp` の `mEntries` / `mApplication`
- 関連 issue: [[0008-bug-fix-non-owning-reference-lifetime]] / [[0013-bug-fix-as-audio-as-video]]
