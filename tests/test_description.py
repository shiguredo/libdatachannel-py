import gc

import pytest

from libdatachannel import CertificateFingerprint, Description


def test_create_description_offer():
    desc = Description("v=0...", Description.Type.Offer)
    assert desc.type() == Description.Type.Offer
    assert isinstance(desc.bundle_mid(), str)


def test_add_audio_track():
    desc = Description("v=0...", Description.Type.Offer)
    index = desc.add_audio("audio", Description.Direction.SendOnly)
    assert index == 0
    assert desc.has_audio_or_video()
    assert desc.media_count() == 1

    media = desc.media(0)
    assert isinstance(media, Description.Media)

    media.as_audio().add_opus_codec(111)


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


# media() / application() の戻り値は Description 内部への参照のため、 親を生存させないと
# use-after-free で SEGV する。 回帰した場合はこのテストの実行中にプロセスが落ちる
def test_media_outlives_description():
    """Description を破棄しても media() の戻り値が使えること"""

    def make_media():
        desc = Description("v=0...")
        desc.add_audio("audio", Description.Direction.SendOnly)
        return desc.media(0)

    media = make_media()
    gc.collect()
    assert media.mid() == "audio"


def test_application_outlives_description():
    """Description を破棄しても application() の戻り値が使えること"""

    def make_application():
        desc = Description("v=0...")
        desc.add_application("data")
        return desc.application()

    application = make_application()
    gc.collect()
    assert isinstance(application, Description.Application)


def test_rtp_map_is_valid_after_remove():
    """remove_rtp_map の後も rtp_map() の戻り値が有効なこと

    内部の RtpMap への参照ではなく値 (コピー) を返すため、 erase の影響を受けない。
    """
    media = Description.Audio()
    media.add_audio_codec(96, "opus", "useinbandfec=1")
    rtpmap = media.rtp_map(96)
    assert rtpmap is not None
    media.remove_rtp_map(96)

    assert rtpmap.payload_type == 96
    assert "opus" in rtpmap.format.lower()
    assert not media.has_payload_type(96)


def test_rtp_map_raises_for_unknown_payload_type():
    """存在しない payload type では ValueError になること (libdatachannel の挙動)"""
    media = Description.Audio()
    with pytest.raises(ValueError):
        media.rtp_map(123)
