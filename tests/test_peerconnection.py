from libdatachannel import (
    DataChannel,
    DataChannelInit,
    Description,
    LocalDescriptionInit,
    PeerConnection,
    Reliability,
)


def test_data_channel_init():
    init = DataChannelInit()
    assert isinstance(init.reliability, Reliability)
    assert init.negotiated is False
    assert init.id is None
    assert init.protocol == ""
    init = DataChannelInit()
    init.reliability.unordered = True
    init.negotiated = True
    init.id = 7
    init.protocol = "webrtc-chat"
    assert init.reliability.unordered is True
    assert init.negotiated is True
    assert init.id == 7
    assert init.protocol == "webrtc-chat"


def test_local_description_init():
    init = LocalDescriptionInit()
    assert init.ice_ufrag is None
    assert init.ice_pwd is None
    init = LocalDescriptionInit()
    init.ice_ufrag = "abc123"
    init.ice_pwd = "xyz456"
    assert init.ice_ufrag == "abc123"
    assert init.ice_pwd == "xyz456"


def test_peerconnection_construction():
    pc = PeerConnection()
    assert pc.state() is PeerConnection.State.New
    assert pc.ice_state() is PeerConnection.IceState.New
    assert pc.gathering_state() is PeerConnection.GatheringState.New
    assert pc.signaling_state() is PeerConnection.SignalingState.Stable


def test_set_local_description():
    pc = PeerConnection()
    dc = pc.create_data_channel("chat")
    assert isinstance(dc, DataChannel)
    assert dc.label() == "chat"
    desc = pc.local_description()
    assert isinstance(desc, Description)
    assert "UDP/DTLS/SCTP" in str(desc)
    assert "sctp-port" in str(desc)
    assert "max-message-size" in str(desc)
