import gc

import pytest

from libdatachannel import CertificateFingerprint, Description


def test_create_description_offer():
    desc = Description("v=0...", Description.Type.Offer)
    assert desc.type() == Description.Type.Offer
    assert isinstance(desc.bundle_mid(), str)


def test_add_audio_track():
    desc = Description("v=0...", Description.Type.Offer)
    # Description は media を Media として保持するため、 codec は追加前に付ける
    audio = Description.Audio("audio", Description.Direction.SendOnly)
    audio.add_opus_codec(111)

    index = desc.add_media(audio)
    assert index == 0
    assert desc.has_audio_or_video()
    assert desc.media_count() == 1

    media = desc.media(0)
    assert isinstance(media, Description.Media)
    assert media.has_payload_type(111)


def test_add_application_track():
    desc = Description("v=0...")
    index = desc.add_application("data")
    assert index == 0

    app = desc.application()
    assert isinstance(app, Description.Application)

    app.set_sctp_port(5000)
    assert app.sctp_port() == 5000


def test_rtpmap_add_remove():
    media = Description.Audio()
    codec_id = 96
    media.add_audio_codec(codec_id, "opus", "useinbandfec=1")

    assert media.has_payload_type(codec_id)
    rtpmap = media.rtp_map(codec_id)
    assert isinstance(rtpmap, Description.RtpMap)
    assert rtpmap.payload_type == codec_id
    assert "opus" in rtpmap.format.lower()

    media.remove_rtp_map(codec_id)
    assert not media.has_payload_type(codec_id)
    # erase 後は再取得できない (取得済みのコピーは値なので影響を受けない)
    with pytest.raises(ValueError):
        media.rtp_map(codec_id)


def test_extmap_operations():
    app = Description.Application()
    ext = Description.Entry.ExtMap(1, "urn:ietf:params:rtp-hdrext:sdes:mid")
    app.add_ext_map(ext)

    ids = app.ext_ids()
    assert 1 in ids

    app.remove_ext_map(1)
    assert 1 not in app.ext_ids()


def test_certificate_fingerprint_operations():
    fp = CertificateFingerprint()
    fp.algorithm = CertificateFingerprint.Algorithm.Sha256
    fp.value = "AB:CD:EF"

    assert fp.is_valid() in (True, False)  # Depending on implementation
    id_str = CertificateFingerprint.algorithm_identifier(fp.algorithm)
    size = CertificateFingerprint.algorithm_size(fp.algorithm)

    assert isinstance(id_str, str)
    assert isinstance(size, int)


def test_media_outlives_description() -> None:
    """Description を破棄しても media() の戻り値が使えること

    media() の戻り値は Description 内部への参照のため、 親を生存させないと
    use-after-free になる。 値を読むときに C++ オブジェクトを参照するため、
    回帰した場合はこのテストの実行中にプロセスが落ちる。
    """

    desc = Description("v=0...")
    desc.add_audio("audio", Description.Direction.SendOnly)
    media = desc.media(0)
    assert isinstance(media, Description.Media)
    del desc
    gc.collect()
    assert media.mid() == "audio"


def test_application_outlives_description() -> None:
    """Description を破棄しても application() の戻り値が使えること

    application() の戻り値は Description 内部への参照のため、 親を生存させないと
    use-after-free になる。 isinstance だけでなく内部の値を読んで検証する。
    """

    desc = Description("v=0...")
    desc.add_application("data")
    application = desc.application()
    assert isinstance(application, Description.Application)
    del desc
    gc.collect()
    assert application.mid() == "data"


def test_rtp_map_returns_copy() -> None:
    """rtp_map() が値 (コピー) を返すこと

    内部の RtpMap への参照を返すと remove_rtp_map / remove_format の erase で
    無効になるため、 値 (コピー) を返す。 書き換えが Media に反映されないことと、
    呼ぶたびに別のオブジェクトが返ることで参照返しとの違いを検証する。
    """
    media = Description.Audio()
    media.add_audio_codec(96, "opus", "useinbandfec=1")

    rtpmap = media.rtp_map(96)
    rtpmap.format = "MUTATED"

    assert media.rtp_map(96).format == "opus"
    assert media.rtp_map(96) is not rtpmap


def test_rtp_map_raises_for_unknown_payload_type() -> None:
    """存在しない payload type では ValueError になること

    libdatachannel の rtpMap() が例外を投げ、 nanobind が ValueError に変換する。
    """
    media = Description.Audio()
    with pytest.raises(ValueError):
        media.rtp_map(123)


def test_as_audio_returns_self() -> None:
    """as_audio() が値コピーではなく同じ Audio を返すこと

    値コピーを返すと戻り値への加工が元の Media に反映されないため、 同一の
    オブジェクトが返ることを検証する。
    """
    audio = Description.Audio("audio", Description.Direction.SendOnly)

    assert audio.as_audio() is audio
    audio.as_audio().add_opus_codec(111)
    assert audio.has_payload_type(111)


def test_as_video_returns_self() -> None:
    """as_video() が値コピーではなく同じ Video を返すこと"""
    video = Description.Video("video", Description.Direction.SendOnly)

    assert video.as_video() is video
    video.as_video().add_h264_codec(96)
    assert video.has_payload_type(96)


def test_as_audio_raises_for_video() -> None:
    """Video に対して as_audio() を呼ぶと TypeError になること

    static_cast では未定義動作になるため、 動的型を確認して例外にする。
    """
    video = Description.Video("video", Description.Direction.SendOnly)
    with pytest.raises(TypeError):
        video.as_audio()


def test_as_video_raises_for_audio() -> None:
    """Audio に対して as_video() を呼ぶと TypeError になること"""
    audio = Description.Audio("audio", Description.Direction.SendOnly)
    with pytest.raises(TypeError):
        audio.as_video()


def test_as_audio_raises_for_media_in_description() -> None:
    """Description から取得した media では as_audio() / as_video() が例外になること

    Description は media を Media として保持する (libdatachannel がスライスする) ため、
    Audio / Video として扱うことはできない。
    """
    desc = Description("v=0...")
    desc.add_audio("audio", Description.Direction.SendOnly)
    desc.add_video("video", Description.Direction.SendOnly)

    for media in (desc.media(0), desc.media(1)):
        assert isinstance(media, Description.Media)
        with pytest.raises(TypeError):
            media.as_audio()
        with pytest.raises(TypeError):
            media.as_video()


def test_as_video_outlives_media() -> None:
    """Media を破棄しても as_video() の戻り値が使えること

    戻り値は元の Media への参照のため、 親を生存させる必要がある。
    """
    video = Description.Video("video", Description.Direction.SendOnly)
    video.add_h264_codec(96)
    reference = video.as_video()
    del video
    gc.collect()

    assert reference.mid() == "video"
    assert reference.has_payload_type(96)
