"""生成される型スタブのエイリアスのテスト

__init__.py が定義するエイリアスが __init__.pyi にも含まれることを検証する。
型チェッカーは __init__.py より __init__.pyi を優先するため、 スタブ側に無いと
利用者の import が型チェックで失敗する (実行時は成功するため気づきにくい)。
"""

from pathlib import Path

import pytest

# 実行時に同じクラスを指すことの確認と、 型チェッカーがこの import を解決できることの
# 確認を兼ねる (上の import が通ること自体がスタブの検証になる)
from libdatachannel import (
    AACRtpDepacketizer,
    AACRtpPacketizer,
    G722RtpDepacketizer,
    G722RtpPacketizer,
    OpusRtpDepacketizer,
    OpusRtpPacketizer,
    PCMARtpDepacketizer,
    PCMARtpPacketizer,
    PCMURtpDepacketizer,
    PCMURtpPacketizer,
)

# スタブに含まれているべきエイリアス (エイリアス名, 参照先)
_STUB_ALIASES = (
    ("AACRtpPacketizer", "OpusRtpPacketizer"),
    ("PCMURtpPacketizer", "PCMARtpPacketizer"),
    ("G722RtpPacketizer", "PCMARtpPacketizer"),
    ("AACRtpDepacketizer", "OpusRtpDepacketizer"),
    ("PCMURtpDepacketizer", "PCMARtpDepacketizer"),
    ("G722RtpDepacketizer", "PCMARtpDepacketizer"),
)


def test_aliases_are_the_same_classes() -> None:
    """エイリアスが実行時に同じクラスを指していること

    エイリアスは __init__.py の代入で作られるため、 同じ型であることを確かめる。
    """
    assert AACRtpPacketizer is OpusRtpPacketizer
    assert PCMURtpPacketizer is PCMARtpPacketizer
    assert G722RtpPacketizer is PCMARtpPacketizer
    assert AACRtpDepacketizer is OpusRtpDepacketizer
    assert PCMURtpDepacketizer is PCMARtpDepacketizer
    assert G722RtpDepacketizer is PCMARtpDepacketizer


def test_stub_contains_aliases() -> None:
    """インストールされた __init__.pyi にエイリアスが含まれること

    wheel に入るのは生成されたスタブなので、 追記が漏れると型チェッカーだけが失敗する。
    生成物の中身を直接確認する。
    """
    import libdatachannel

    stub = Path(libdatachannel.__file__).with_name("__init__.pyi")
    if not stub.exists():
        pytest.skip("型スタブがインストールされていない")

    content = stub.read_text()
    for name, alias in _STUB_ALIASES:
        assert f"{name} = {alias}" in content, f"{name} が型スタブに無い"
