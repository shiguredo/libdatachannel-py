# as_audio() と as_video() が値コピーを返し変更加工が消失する

- Priority: High
- Created: 2026-08-30
- Completed: {YYYY-MM-DD}
- Branch: feature/fix-as-audio-as-video
- Polished: {YYYY-MM-DD}

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
media.has_payload_type(97)  # False → 加工が消失している
```

- `bind_description` の as_audio / as_video は `*static_cast<Description::Audio*>(p)` のように値を返す
- libdatachannel の Description::Audio / Description::Video は Description::Media を継承する (`include/rtc/description.hpp`)。Media そのもののインスタンスを兄弟クラスに static_cast するのは未定義動作
- 値返しのため nanobind は新しい C++ オブジェクト (コピー) を生成し、加工が元に反映されない

## 設計方針

- 値返しをやめる。動的型を dynamic_cast で検証し、正しい型であれば参照を返す (keep_alive 付き)。動的型が異なる場合は `nb::type_error` を投げる
- 修正後、tests/test_description.py の as_audio 経路を「元の media に反映される」ことを検証する形に更新する

## 完了条件

- 動的型と異なる as_audio / as_video の呼び出しが例外になること (未定義動作に到達しないこと)
- as_audio / as_video の戻り値への加工が元の Media に反映されること (テストで検証)
- `uv sync && make test` で全テストが PASS すること
- `/review-diff-code` の致命的 / 重要指摘が 0 件であること

## 参考

- 対象シンボル: `bind_description` 内の as_audio / as_video (src/bind_libdatachannel.cpp)
- libdatachannel v0.24.0: `include/rtc/description.hpp` (`Description::Audio`、`Description::Video`)
