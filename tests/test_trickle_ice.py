"""examples/trickle_ice.py のテスト

RFC 9725 Figure 3 の PATCH body と同じ形の fragment が組み立てられることを確認する。
"""

import importlib.util
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parent.parent / "examples" / "trickle_ice.py"
_SPEC = importlib.util.spec_from_file_location("trickle_ice", _MODULE_PATH)
assert _SPEC is not None
trickle_ice = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(trickle_ice)

build_sdp_fragment = trickle_ice.build_sdp_fragment

# RFC 9725 Figure 3 の offer 相当 (bundle グループと audio の m= セクション)
_OFFER = (
    "v=0\r\n"
    "o=- 0 0 IN IP4 127.0.0.1\r\n"
    "s=-\r\n"
    "t=0 0\r\n"
    "a=group:BUNDLE 0 1\r\n"
    "m=audio 9 UDP/TLS/RTP/SAVPF 111\r\n"
    "a=mid:0\r\n"
    "a=ice-ufrag:EsAw\r\n"
    "a=ice-pwd:P2uYro0UCOQ4zxjKXaWCBui1\r\n"
    "a=setup:actpass\r\n"
    "m=video 9 UDP/TLS/RTP/SAVPF 96\r\n"
    "a=mid:1\r\n"
    "a=ice-ufrag:EsAw\r\n"
    "a=ice-pwd:P2uYro0UCOQ4zxjKXaWCBui1\r\n"
)

# RFC 9725 Figure 3 の PATCH body
_EXPECTED = (
    "a=group:BUNDLE 0 1\r\n"
    "m=audio 9 UDP/TLS/RTP/SAVPF 111\r\n"
    "a=mid:0\r\n"
    "a=ice-ufrag:EsAw\r\n"
    "a=ice-pwd:P2uYro0UCOQ4zxjKXaWCBui1\r\n"
    "a=candidate:1387637174 1 udp 2122260223 192.0.2.1 61764 typ host generation 0 ufrag EsAw network-id 1\r\n"
    "a=candidate:3471623853 1 udp 2122194687 198.51.100.2 61765 typ host generation 0 ufrag EsAw network-id 2\r\n"
    "a=end-of-candidates\r\n"
)


def test_build_sdp_fragment_matches_rfc9725_figure3() -> None:
    """RFC 9725 Figure 3 と同じ形の fragment になること

    bundle ポリシーでは offerer-tagged の m= 行 (audio) のみを含め、 candidate は
    a=candidate 行として並べ、 最後に a=end-of-candidates を付ける。
    """
    candidates = [
        "candidate:1387637174 1 udp 2122260223 192.0.2.1 61764 typ host generation 0 ufrag EsAw network-id 1",
        "candidate:3471623853 1 udp 2122194687 198.51.100.2 61765 typ host generation 0 ufrag EsAw network-id 2",
    ]

    assert build_sdp_fragment(_OFFER, candidates) == _EXPECTED


def test_build_sdp_fragment_without_candidates() -> None:
    """candidate が無い場合も fragment として成立すること

    a=end-of-candidates だけを送る形になる (candidate が集まらなかった場合)。
    """
    fragment = build_sdp_fragment(_OFFER, [])

    assert "a=end-of-candidates" in fragment
    assert "a=candidate:" not in fragment
    assert "m=video" not in fragment


def test_build_sdp_fragment_without_end_of_candidates() -> None:
    """収集が終わっていない場合は a=end-of-candidates を付けないこと

    gathering が完了する前に付けると、 対向の ICE が早期に完了し得る。
    """
    fragment = build_sdp_fragment(_OFFER, ["candidate:1 1 udp 1 192.0.2.1 1 typ host"], False)

    assert "a=candidate:1 1 udp 1 192.0.2.1 1 typ host\r\n" in fragment
    assert "a=end-of-candidates" not in fragment


def test_build_sdp_fragment_accepts_attribute_form() -> None:
    """a=candidate: の形で渡された candidate もそのまま扱えること"""
    fragment = build_sdp_fragment(_OFFER, ["a=candidate:1 1 udp 1 192.0.2.1 1 typ host"])

    assert "a=candidate:1 1 udp 1 192.0.2.1 1 typ host\r\n" in fragment


def test_build_sdp_fragment_requires_ice_credentials() -> None:
    """ICE 資格情報が無い SDP では ValueError になること"""
    with pytest.raises(ValueError):
        build_sdp_fragment("v=0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\na=mid:0\r\n", [])


def test_build_sdp_fragment_requires_media_section() -> None:
    """m= 行が無い SDP では ValueError になること"""
    with pytest.raises(ValueError):
        build_sdp_fragment("v=0\r\na=ice-ufrag:x\r\na=ice-pwd:y\r\n", [])
