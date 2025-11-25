import sys
import time

import pytest

from libdatachannel import (
    Candidate,
    Configuration,
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


# https://github.com/paullouisageneau/libdatachannel/blob/0e40aeb058b947014a918a448ce2d346e6ab14fe/test/track.cpp
# を Python に直したもの
def test_track():
    config1 = Configuration()
    pc1 = PeerConnection(config1)

    config2 = Configuration()
    config2.port_range_begin = 5000
    config2.port_range_end = 6000
    pc2 = PeerConnection(config2)

    def pc1_on_local_description(desc):
        print("Description 1: " + str(desc))
        pc2.set_remote_description(Description(str(desc)))

    def pc1_on_local_candidate(candidate):
        print("Candidate 1: " + str(candidate))
        pc2.add_remote_candidate(Candidate(str(candidate)))

    def pc1_on_state_change(state):
        print("State 1: " + str(state))

    def pc1_on_gathering_state_change(state):
        print("Gathering state 1: " + str(state))

    pc1.on_local_description(pc1_on_local_description)
    pc1.on_local_candidate(pc1_on_local_candidate)
    pc1.on_state_change(pc1_on_state_change)
    pc1.on_gathering_state_change(pc1_on_gathering_state_change)

    def pc2_on_local_description(desc):
        print("Description 2: " + str(desc))
        pc1.set_remote_description(Description(str(desc)))

    def pc2_on_local_candidate(candidate):
        print("Candidate 2: " + str(candidate))
        pc1.add_remote_candidate(Candidate(str(candidate)))

    def pc2_on_state_change(state):
        print("State 2: " + str(state))

    def pc2_on_gathering_state_change(state):
        print("Gathering state 2: " + str(state))

    pc2.on_local_description(pc2_on_local_description)
    pc2.on_local_candidate(pc2_on_local_candidate)
    pc2.on_state_change(pc2_on_state_change)
    pc2.on_gathering_state_change(pc2_on_gathering_state_change)

    t2 = None
    new_track_mid = ""

    def pc2_on_track(t):
        nonlocal t2
        mid = t.mid()
        print(f'Track 2: Received track with mid "{mid}"')
        if mid != new_track_mid:
            print("Wrong track mid", file=sys.stderr)
            return

        def t_on_open():
            print(f'Track 2: Track with mid "{mid}" is open')

        def t_on_closed():
            print(f'Track 2: Track with mid "{mid}" is closed')

        t.on_open(t_on_open)
        t.on_closed(t_on_closed)
        t2 = t

    pc2.on_track(pc2_on_track)

    # Test opening a track
    new_track_mid = "test"

    media = Description.Video(new_track_mid, Description.Direction.SendOnly)
    media.add_h264_codec(96)
    media.set_bitrate(3000)
    media.add_ssrc(1234, "video-send")

    media_sdp1 = str(media)
    media_sdp2 = str(Description.Media(media_sdp1))
    assert media_sdp1 == media_sdp2

    t1 = pc1.add_track(media)

    pc1.set_local_description()

    attempts = 10
    while (not t1.is_open() or t2 is None or not t2.is_open()) and attempts > 0:
        attempts -= 1
        time.sleep(1)

    assert pc1.state() == PeerConnection.State.Connected
    assert pc2.state() == PeerConnection.State.Connected

    assert t1.is_open()
    assert t2 is not None
    assert t2.is_open()

    # Test renegotiation
    new_track_mid = "added"

    media2 = Description.Video(new_track_mid, Description.Direction.SendOnly)
    media2.add_h264_codec(96)
    media2.set_bitrate(3000)
    media2.add_ssrc(2468, "video-send")

    # NOTE: Overwriting the old shared_ptr for t1 will cause it's respective
    #       track to be dropped (so it's SSRCs won't be on the description next time)
    t1 = pc1.add_track(media2)

    t2 = None
    pc1.set_local_description()

    attempts = 10
    while (not t1.is_open() or t2 is None or not t2.is_open()) and attempts > 0:
        attempts -= 1
        time.sleep(1)

    assert t1.is_open()
    assert t2 is not None
    assert t2.is_open()

    # Delay close of peer 2 to check closing works properly
    pc1.close()
    time.sleep(1)
    pc2.close()
    time.sleep(1)

    assert t1.is_closed()
    assert t2.is_closed()

    print("Success")


# test_track() と同じようなことをするけど、バインドしたまま close せずに終了する
@pytest.mark.skip(reason="なぜかブロックして止まったままになるのでいったんスキップ")
def test_leak():
    config1 = Configuration()
    pc1 = PeerConnection(config1)

    config2 = Configuration()
    config2.port_range_begin = 5000
    config2.port_range_end = 6000
    pc2 = PeerConnection(config2)

    def pc1_on_local_description(desc):
        print("Description 1: " + str(desc))
        pc2.set_remote_description(Description(str(desc)))

    def pc1_on_local_candidate(candidate):
        print("Candidate 1: " + str(candidate))
        pc2.add_remote_candidate(Candidate(str(candidate)))

    def pc1_on_state_change(state):
        print("State 1: " + str(state))

    def pc1_on_gathering_state_change(state):
        print("Gathering state 1: " + str(state))

    pc1.on_local_description(pc1_on_local_description)
    pc1.on_local_candidate(pc1_on_local_candidate)
    pc1.on_state_change(pc1_on_state_change)
    pc1.on_gathering_state_change(pc1_on_gathering_state_change)

    def pc2_on_local_description(desc):
        print("Description 2: " + str(desc))
        pc1.set_remote_description(Description(str(desc)))

    def pc2_on_local_candidate(candidate):
        print("Candidate 2: " + str(candidate))
        pc1.add_remote_candidate(Candidate(str(candidate)))

    def pc2_on_state_change(state):
        print("State 2: " + str(state))

    def pc2_on_gathering_state_change(state):
        print("Gathering state 2: " + str(state))

    pc2.on_local_description(pc2_on_local_description)
    pc2.on_local_candidate(pc2_on_local_candidate)
    pc2.on_state_change(pc2_on_state_change)
    pc2.on_gathering_state_change(pc2_on_gathering_state_change)

    t2 = None
    new_track_mid = ""

    def pc2_on_track(t):
        nonlocal t2
        mid = t.mid()
        print(f'Track 2: Received track with mid "{mid}"')
        if mid != new_track_mid:
            print("Wrong track mid", file=sys.stderr)
            return

        def t_on_open():
            print(f'Track 2: Track with mid "{mid}" is open')

        def t_on_closed():
            print(f'Track 2: Track with mid "{mid}" is closed')

        t.on_open(t_on_open)
        t.on_closed(t_on_closed)
        t2 = t

    pc2.on_track(pc2_on_track)

    # Test opening a track
    new_track_mid = "test"

    media = Description.Video(new_track_mid, Description.Direction.SendOnly)
    media.add_h264_codec(96)
    media.set_bitrate(3000)
    media.add_ssrc(1234, "video-send")

    media_sdp1 = str(media)
    media_sdp2 = str(Description.Media(media_sdp1))
    assert media_sdp1 == media_sdp2

    t1 = pc1.add_track(media)

    pc1.set_local_description()

    attempts = 10
    while (not t1.is_open() or t2 is None or not t2.is_open()) and attempts > 0:
        attempts -= 1
        time.sleep(1)

    assert pc1.state() == PeerConnection.State.Connected
    assert pc2.state() == PeerConnection.State.Connected

    assert t1.is_open()
    assert t2 is not None
    assert t2.is_open()

    # これが無いとリークする
    pc1 = None
    pc2 = None
