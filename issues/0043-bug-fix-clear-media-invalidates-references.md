# clear_media() などが media() / application() の取得済み参照を無効にする

- Priority: High
- Created: 2026-10-09
- Completed: {YYYY-MM-DD}
- Branch: feature/fix-clear-media-invalidates-references
- Polished: 2026-10-10

## 目的

`Description` には、 それ以前に取得した `media()` / `application()` の戻り値を無効にする操作がある (`clear_media()` は `mEntries.clear()` / `mApplication.reset()`、 `add_media(Application)` / `add_application()` は `removeApplication()`)。 戻り値は内部オブジェクトへの生ポインタのため、 無効化後に触ると use-after-free でプロセスが落ちるか、 別オブジェクトを指す。 親を生存させるだけでは防げないため、 Python から踏めるクラッシュ経路を塞ぐ。

## 優先度根拠

- 実測 ([[0008-bug-fix-non-owning-reference-lifetime]] の修正後ビルド): `desc.add_audio("audio", ...)` → `m = desc.media(0)` → `desc.clear_media()` → `m.mid()` で exit 139 (SIGSEGV)
- [[0008-bug-fix-non-owning-reference-lifetime]] で親の寿命に紐付ける修正を入れたが、 この経路は解放の主体が親ではなく破壊的操作のため解消していない
- `clear_media()` は SDP を組み直す際に使われ得る操作であり、 取得済みの `Media` を保持したまま呼ぶコードは自然に書ける
- このリポジトリには `clear_media()` を使っている箇所が無い (examples / tests / README / docs で 0 件、 binding の定義だけ)。 一方、 `media()` / `application()` の戻り値を書き換える使い方は `tests/test_description.py` に実在する (実測)

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
- `add_media(Application)` / `add_application()` も `removeApplication()` を先に呼ぶため、 取得済みの `application()` 参照を無効にする (実測: `add_media(Description.Application("data2"))` の後、 旧参照の `mid()` が `data` から `data2` に変わり、 新しい参照と同じオブジェクトを指す)
- `add_media(Media)` / `add_video()` / `add_audio()` は `mEntries` (`vector<shared_ptr<Entry>>`) に追加するだけで、 取得済みの `Media*` を無効にしない (実測)。 `add_rtp_map()` / `remove_rtp_map()` / `remove_format()` は `Description` ではなく `Description.Media` のメソッドで、 `Media` 自身の状態だけを変える

## 設計方針

- 案 A (`media()` / `application()` を値 (コピー) で返す) を採用する。 クラッシュ経路を塞ぐことを優先し、 かつ公開 API を減らさないためである
  - `media(int)` は `variant<Media*, Application*>` を返すため、 binding 側で型を見て `Media` / `Application` のコピーを作って返す。 Python 側の型は `object` のまま (動的な型は変わらない)。 0008 で `rtp_map()` を値返しにしたのと同じ考え方である (`as_audio()` / `as_video()` は 0013 で参照返しになったため、 そことは方向が違う)
  - `application()` も同じくコピーを返す
- この変更で、 戻り値を書き換えても `Description` に反映されなくなる。 同じ SDP を作るには、 codec などを足した media を組み立ててから `add_media()` する。 これは後方互換のない変更なので `[CHANGE]` として記録し、 この制約と組み立て方を docstring と `CHANGES.md` に明記する
  - `add_rtp_map()` / `remove_rtp_map()` / `remove_format()` は `Description` ではなく `Description.Media` のメソッドであるため、 `Description` に追加済みの media へ後から codec を足しても `Description` に反映されなくなる (コピーへの追加になる)。 `tests/test_description.py` にある「`desc.media(0)` の戻り値に `add_rtp_map()` する」テストは、 `Description.Video` / `Description.Audio` を作って codec を足してから `add_media()` する形に置き換える
- 案 B (取得済みの `Entry` への `shared_ptr` を Python 側で保持する) はこのリポジトリでは実装できない。 `Description` の `mEntries` / `mApplication` は private で `shared_ptr` を返す公開 API が無く、 `_deps` の libdatachannel に patch を当てる仕組みも無いためである。 恒久的には libdatachannel 側に `shared_ptr` を返す API を足すのが本筋なので、 upstream への提案として記録する (この issue では対応しない)
- 案 C (`clear_media()` が取得済み参照を無効にすると明記するだけ) は、 SIGSEGV の経路を残すため採らない
- 無効化する操作 (`clear_media()` / `add_media(Application)` / `add_application()`) を binding から削除する案は、 SDP を組み直す機能を失うため採らない
- 採用した案で、 破壊的操作のあとに古いハンドルを触っても落ちないことをテストで固定する (案 A ではコピーになるため常に安全である)

## 完了条件

- `media()` / `application()` がコピーを返すこと (そのため `clear_media()` / `add_media(Application)` / `add_application()` のあとでも安全であること)、 戻り値の書き換えが `Description` に反映されないこと、 `Description` に追加済みの media へ後から codec を足しても反映されないこと (コピーへの追加になること) が、 docstring と `CHANGES.md` に明記されていること (書き換えの代わりに media を組み立ててから `add_media()` する手順も書く)
- `clear_media()` / `add_media(Application)` / `add_application()` のあとに、 それ以前に取得したハンドルを触っても SIGSEGV しないこと (テストで固定する)
- `clear_media()` / `add_media(Application)` / `add_application()` のあとに、 それ以前に取得したハンドルが元の値 (`mid()` など) を返し続けること
- `Description.RtpMap` を作って `Description.Video` / `Description.Audio` の `add_rtp_map()` に渡してから `add_media()` する組み立て方で、 `tests/test_description.py` の該当テストと同じ media 部分の SDP が得られること (テストをその形に置き換える。 `o=rtc <session id>` は毎回変わるため全体の文字列一致は求めない)
- C++ の変更を含むため `make develop` で再ビルドし、 `src/libdatachannel/__init__.pyi` のスタブを再生成したうえで、 `prek run --all-files pytest` が PASS すること
- CI (wheel.yml の leg / prek.yml の `ty` ジョブ) が PASS すること
- `CHANGES.md` の `## develop` に変更内容が記録されていること (`[CHANGE]` として記録する)
- `CHANGES.md` の `## develop` にある 0013 の `[CHANGE]` エントリにある「既存の media には … `add_rtp_map()` へ渡して追加する」という案内と、 binding の `as_audio()` / `as_video()` の docstring、 `get_media` のコメント、 `tests/test_description.py` の参照の寿命に関するテストの docstring を、 コピーを返す挙動に合わせて修正すること
- libdatachannel に `shared_ptr` を返す API を足す提案の内容を、 解決方法に記録すること (この issue では upstream への提案までとし、 実装はしない)
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること

## スコープ外 (関連する未解決問題)

- `remove_format` / `remove_rtp_map` による `rtp_map()` の無効化は [[0008-bug-fix-non-owning-reference-lifetime]] の値返しで解消済み
- `as_audio()` / `as_video()` の値コピーは [[0013-bug-fix-as-audio-as-video]] で解決済み (同じオブジェクトを返す参照返しになった)

## 参考

- libdatachannel v0.24.0: `source/src/description.cpp` の `clearMedia()` / `removeApplication()` / `addMedia()` / `removeRtpMap()` / `removeFormat()`、 `source/include/rtc/description.hpp` の `mEntries` / `mApplication` (どちらも private で、 `media(int)` は `variant<Media*, Application*>` を返す)
- 関連 issue: [[0008-bug-fix-non-owning-reference-lifetime]] / [[0013-bug-fix-as-audio-as-video]]
