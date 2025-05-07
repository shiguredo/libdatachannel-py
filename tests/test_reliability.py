from datetime import timedelta

import libdatachannel


def test_reliability():
    rel = libdatachannel.Reliability()
    assert rel.unordered is False
    assert rel.max_packet_lifetime is None
    assert rel.max_retransmits is None
    rel.unordered = True
    rel.max_packet_lifetime = timedelta(milliseconds=250)
    rel.max_retransmits = 3
    assert rel.unordered is True
    assert rel.max_packet_lifetime.total_seconds() == 0.25
    assert rel.max_retransmits == 3
