from datetime import timedelta

from libdatachannel import (
    CertificateType,
    Configuration,
    IceServer,
    ProxyServer,
    TransportPolicy,
    WebSocketConfiguration,
    WebSocketServerConfiguration,
)


def test_ice_server_stun_url_only():
    server = IceServer("stun:example.com")
    assert server.hostname == "example.com"
    assert server.type == IceServer.Type.Stun


def test_ice_server_stun_hostname_port():
    server = IceServer("stun.example.com", 3478)
    assert server.hostname == "stun.example.com"
    assert server.port == 3478


def test_ice_server_turn_full():
    server = IceServer("turn.example.com", 3478, "user", "pass", IceServer.RelayType.TurnTcp)
    assert server.username == "user"
    assert server.password == "pass"
    assert server.relay_type == IceServer.RelayType.TurnTcp


def test_proxy_server_basic():
    proxy = ProxyServer("http://proxy.example.com")
    assert proxy.hostname == "proxy.example.com"


def test_proxy_server_full():
    proxy = ProxyServer(ProxyServer.Type.Socks5, "proxy.example.com", 1080, "alice", "secret")
    assert proxy.type == ProxyServer.Type.Socks5
    assert proxy.username == "alice"
    assert proxy.password == "secret"


def test_configuration_fields():
    config = Configuration()
    config.port_range_begin = 2000
    config.port_range_end = 3000
    config.disable_auto_gathering = True

    assert config.port_range_begin == 2000
    assert config.port_range_end == 3000
    assert config.disable_auto_gathering is True


def test_add_ice_servers_to_configuration():
    stun = IceServer("stun.example.com", 3478)
    turn = IceServer("turn.example.com", 3478, "bob", "pw")

    config = Configuration()
    config.ice_servers = [stun, turn]

    assert len(config.ice_servers) == 2
    assert config.ice_servers[0].hostname == "stun.example.com"
    assert config.ice_servers[1].username == "bob"


def test_websocket_configuration_defaults():
    ws_config = WebSocketConfiguration()
    assert ws_config.disable_tls_verification is False
    assert ws_config.proxy_server is None
    assert isinstance(ws_config.protocols, list)
    assert ws_config.connection_timeout is None
    assert ws_config.ping_interval is None
    assert ws_config.max_outstanding_pings is None
    assert ws_config.max_message_size is None


def test_websocket_configuration_fields():
    ws_config = WebSocketConfiguration()
    ws_config.disable_tls_verification = True
    ws_config.protocols = ["json", "binary"]
    ws_config.connection_timeout = timedelta(seconds=10)
    ws_config.ping_interval = timedelta(seconds=30)
    ws_config.max_outstanding_pings = 5
    ws_config.ca_certificate_pem_file = "/path/ca.pem"
    ws_config.certificate_pem_file = "/path/cert.pem"
    ws_config.key_pem_file = "/path/key.pem"
    ws_config.key_pem_pass = "pass123"
    ws_config.max_message_size = 1048576

    assert ws_config.disable_tls_verification is True
    assert ws_config.protocols == ["json", "binary"]
    assert ws_config.connection_timeout.total_seconds() == 10
    assert ws_config.ping_interval.total_seconds() == 30
    assert ws_config.max_outstanding_pings == 5
    assert ws_config.ca_certificate_pem_file == "/path/ca.pem"
    assert ws_config.max_message_size == 1048576


def test_websocket_server_configuration_defaults():
    ws_server = WebSocketServerConfiguration()
    assert ws_server.port == 8080
    assert ws_server.enable_tls is False
    assert ws_server.certificate_pem_file is None
    assert ws_server.key_pem_file is None
    assert ws_server.key_pem_pass is None
    assert ws_server.bind_address is None
    assert ws_server.connection_timeout is None
    assert ws_server.max_message_size is None


def test_websocket_server_configuration_custom():
    ws_server = WebSocketServerConfiguration()
    ws_server.port = 443
    ws_server.enable_tls = True
    ws_server.certificate_pem_file = "/tls/cert.pem"
    ws_server.key_pem_file = "/tls/key.pem"
    ws_server.key_pem_pass = "secret"
    ws_server.bind_address = "0.0.0.0"
    ws_server.connection_timeout = timedelta(seconds=5)
    ws_server.max_message_size = 65536

    assert ws_server.port == 443
    assert ws_server.enable_tls is True
    assert ws_server.key_pem_pass == "secret"
    assert ws_server.connection_timeout.total_seconds() == 5
    assert ws_server.max_message_size == 65536


def test_configuration_optional_fields():
    config = Configuration()
    config.mtu = 1400
    config.max_message_size = 256000
    config.certificate_pem_file = "/certs/cert.pem"
    config.key_pem_file = "/certs/key.pem"
    config.key_pem_pass = "topsecret"

    assert config.mtu == 1400
    assert config.max_message_size == 256000
    assert config.certificate_pem_file == "/certs/cert.pem"


def test_configuration_enums():
    config = Configuration()
    config.certificate_type = CertificateType.Rsa
    config.ice_transport_policy = TransportPolicy.Relay

    assert config.certificate_type == CertificateType.Rsa
    assert config.ice_transport_policy == TransportPolicy.Relay
