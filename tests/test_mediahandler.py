from libdatachannel import PyMediaHandler, make_message


class DummyHandler(PyMediaHandler):
    def __init__(self):
        super().__init__()
        self.media_called = False
        self.incoming_called = False
        self.outgoing_called = False

    def media(self, desc):
        self.media_called = True

    def incoming(self, messages, send):
        self.incoming_called = True
        assert isinstance(messages, list)
        assert len(messages) == 1
        assert callable(send)
        send(messages[0])

    def outgoing(self, messages, send):
        self.outgoing_called = True
        assert isinstance(messages, list)
        assert len(messages) == 1
        assert callable(send)
        send(messages[0])


def test_incoming_chain():
    h = DummyHandler()

    # メッセージを 1 つ作って渡す
    msg = make_message(10)
    msgs = [msg]

    called = []

    def send_fn(m):
        called.append(m)

    h.incoming_chain(msgs, send_fn)
    assert h.incoming_called is True
    assert len(called) == 1 and called[0] is msg


def test_chaining():
    h1 = DummyHandler()
    h2 = DummyHandler()
    h1.add_to_chain(h2)

    called = []

    def send_fn(m):
        called.append(m)

    h1.incoming_chain([make_message(10)], send_fn)
    assert h1.incoming_called is True
    assert h2.incoming_called is True
    assert len(called) == 2

    h1.incoming_called = False
    h2.incoming_called = False
    called.clear()
    h1.next().incoming_chain([make_message(10)], send_fn)
    assert h1.incoming_called is False
    assert h2.incoming_called is True
    assert len(called) == 1
