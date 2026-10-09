# as_audio() と as_video() が値コピーを返し変更加工が消失する

- Priority: High
- Created: 2026-08-30
- Completed: {YYYY-MM-DD}
- Branch: feature/fix-as-audio-as-video
- Polished: 2026-10-09

## 目的

Description.Media.as_audio() / as_video() は Media を Audio / Video に static_cast して値コピーを返す。そのため (1) 実際の動的型が異なる場合は未定義動作、(2) 動的型が正しくても戻り値への加工が元の Media に反映されない。Media 内の codec を操作する手段として機能していない。

## 優先度根拠

- 実測: as_video() の戻り値に add_video_codec しても元の Media に反映されない (has_payload_type が False のまま)
- `Description.Media(sdp)` で作った Media (Audio でも Video でもない動的型) に対する as_audio() は static_cast の未定義動作
- tests/test_description.py がコピー先への加工で終わっており、この問題を検出できていない

## 現状

再現手順:

```python
from libdatachannel import Description

media = Description.Video("video", Description.Direction.SendOnly)
media.add_h264_codec(96)

audio = media.as_audio()  # 動的型は Video → static_cast の未定義動作
```

```python
media = Description.Video("video", Description.Direction.SendOnly)
media.add_video_codec(96, "h264")
media.as_video().add_video_codec(97, "h265")
assert media.has_payload_type(97)  # False → 加工が消失している
```

さらに、 修正の前提として重要な制約がある。 libdatachannel v0.24.0 は `Description` に
`add_audio()` / `add_video()` / `add_media()` で追加した media も、 SDP を parse した media も、
常に base の `Description::Media` として保持する (`createEntry` は `make_shared<Media>`、
`addMedia(Media)` は値渡しでスライスする)。 つまり `Description` から取得した media の動的型は
常に `Media` であり、 `as_audio()` / `as_video()` が動的型の一致で成功する経路は存在しない。

- `bind_description` の as_audio / as_video は `*static_cast<Description::Audio*>(p)` のように値を返す
- libdatachannel の Description::Audio / Description::Video は Description::Media を継承する (`include/rtc/description.hpp`)。Media そのもののインスタンスを兄弟クラスに static_cast するのは未定義動作
- 値返しのため nanobind は新しい C++ オブジェクト (コピー) を生成し、加工が元に反映されない

## 設計方針

- 値返しをやめる。 `dynamic_cast` で動的型を検証し、 一致した場合は元のオブジェクトへの参照 (`nb::rv_policy::reference_internal`) を返す。 動的型が異なる場合は `nb::type_error` を投げる (static_cast の未定義動作に到達させない)
- ただし `Description` から取得した media の動的型は常に `Media` のため、 この経路は必ず例外になる。 代替を issue と CHANGES に明記する
  - `Description.Audio` / `Description.Video` を直接作って codec を追加し、 その後 `add_media()` する (codec は Media 側に保持されるため引き継がれる)
  - 既に追加済みの media へ codec を足す場合は `add_rtp_map()` に `RtpMap` を渡す
- テストは 2 本立てにする
  - `Description` から取得した media の `as_audio()` / `as_video()` が `TypeError` になること
  - 動的型が `Audio` / `Video` のオブジェクトでは `as_audio()` / `as_video()` が同じオブジェクトを返し、 加工が反映されること

## 完了条件

- 動的型と異なる as_audio / as_video の呼び出しが `TypeError` になること (未定義動作に到達しないこと)
- 動的型が一致する場合は同じオブジェクト (参照) が返り、 加工が元の Media に反映されること
- `Description` から取得した media では例外になること (スライシングの帰結として issue と CHANGES に明記)
- `prek run --all-files pytest` (prek.toml の pytest フック = 既知の恒停テストを `--deselect` で除外) が PASS する
- CI (wheel.yml の leg / prek.yml の `ty` ジョブ) が PASS する
- `CHANGES.md` の `## develop` に変更内容 (`[CHANGE]`) が記録されている
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること

## 参考

- 対象シンボル: `bind_description` 内の as_audio / as_video (src/bind_libdatachannel.cpp)
- libdatachannel v0.24.0: `include/rtc/description.hpp` (`Description::Audio`、`Description::Video`)、 `source/src/description.cpp` (`createEntry` は `make_shared<Media>`、 `addMedia(Media)` は値渡しでスライスする)
- 値コピーを返す他の binding: `reciprocate()` は `Media` を値で返す (新しい Media を作る意図的な仕様のため対象外)
