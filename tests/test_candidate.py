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


def test_candidate_resolution_and_equality():
    c1 = Candidate("candidate:1 1 UDP 2122260223 192.168.0.1 12345 typ host")
    c2 = Candidate("candidate:1 1 UDP 2122260223 192.168.0.1 12345 typ host")
    c3 = Candidate("candidate:2 1 UDP 2122260223 192.168.0.1 12345 typ host")

    assert c1 == c2
    assert c1 != c3

    resolved = c1.resolve()
    # ダミーの IP なので実際に解決できるかどうかは気にしない
    assert isinstance(resolved, bool)
