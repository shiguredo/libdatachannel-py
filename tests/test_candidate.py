from libdatachannel import Candidate


def test_candidate_construction():
    c1 = Candidate()
    c2 = Candidate("candidate:1 1 UDP 2122260223 192.168.0.1 12345 typ host")
    c3 = Candidate("candidate:1 1 UDP 2122260223 192.168.0.1 12345 typ host", "audio")

    assert isinstance(c1, Candidate)
    assert isinstance(c2, Candidate)
    assert isinstance(c3, Candidate)
    assert c3.mid() == "audio"


def test_candidate_attributes_and_conversion():
    c = Candidate("candidate:1 1 UDP 2122260223 192.168.0.1 12345 typ host")
    assert c.candidate() == "candidate:1 1 UDP 2122260223 192.168.0.1 12345 typ host"
    assert str(c) == "a=candidate:1 1 UDP 2122260223 192.168.0.1 12345 typ host"
    assert c.type() is Candidate.Type.Host
    assert c.transport_type() is Candidate.TransportType.Udp


def test_candidate_change_address():
    c = Candidate("candidate:1 1 UDP 2122260223 192.168.0.1 12345 typ host")
    c.change_address("127.0.0.1")
    c.change_address("127.0.0.1", 54321)
    c.change_address("127.0.0.1", "80")


def test_candidate_resolve_returns_bool():
    """Candidate.resolve() が真偽値を返し、 例外を出さないこと

    ダミーの IP なので実際に解決できるかどうかは環境次第のため、 戻り値の型だけを見る
    (等価性は test_equal_candidates_have_same_hash などで検証する)。
    """
    candidate = Candidate("candidate:1 1 UDP 2122260223 192.168.0.1 12345 typ host")

    assert isinstance(candidate.resolve(), bool)


# SDP の candidate 行が同じ 2 つの Candidate
_SDP = "candidate:1 1 UDP 2122260223 192.168.0.1 12345 typ host"


def test_equal_candidates_have_same_hash() -> None:
    """同じ candidate 行の Candidate が == かつ同一 hash であること

    __eq__ を定義したクラスは __hash__ も必要 (a == b なら hash(a) == hash(b))。
    """
    first = Candidate(_SDP)
    second = Candidate(_SDP)

    assert first == second
    assert hash(first) == hash(second)


def test_equal_candidates_are_deduplicated_in_set_and_dict() -> None:
    """dict / set で同じ candidate 行の Candidate が 1 つに畳まれること"""
    first = Candidate(_SDP)
    second = Candidate(_SDP)

    assert len({first, second}) == 1
    assert {first: "value"}.get(second) == "value"


def test_ne_is_the_negation_of_eq() -> None:
    """!= が == の否定になっていること

    libdatachannel の operator!= は foundation のみを比較するため、 同じ foundation で
    node が違う場合に == も != も False になっていた。 __ne__ のバインドを外し、
    Python が __eq__ の否定を導出するようにしている。
    """
    first = Candidate("candidate:1 1 UDP 2122260223 192.168.0.1 12345 typ host")
    other = Candidate("candidate:1 1 UDP 2122260223 192.168.0.2 12345 typ host")

    assert first != other
    assert (first == other) is False
    assert len({first, other}) == 2


def test_equality_compares_candidate_line() -> None:
    """candidate 行が違えば等しくないこと

    priority と type は libdatachannel の operator== では比較されないが、 candidate 行
    としては別の候補であるため、 binding は等しくないものとして扱う。
    """
    base = Candidate("candidate:1 1 UDP 2122260223 192.168.0.1 12345 typ host")

    assert base != Candidate("candidate:1 1 UDP 2122260222 192.168.0.1 12345 typ host")
    assert base != Candidate("candidate:1 1 UDP 2122260223 192.168.0.1 12345 typ srflx")


def test_eq_with_other_types_returns_false() -> None:
    """Candidate 以外と比較したときに False を返すこと

    Python の __eq__ は任意の object と比較され得る。 ただし nanobind の引数変換の
    都合で None との比較は TypeError になる (修正前からの挙動で、 本 issue の範囲外)。
    """
    candidate = Candidate(_SDP)

    assert (candidate == 1) is False
    assert (candidate == "candidate:1 1 UDP 2122260223 192.168.0.1 12345 typ host") is False
