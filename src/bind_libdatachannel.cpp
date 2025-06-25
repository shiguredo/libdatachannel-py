// nanobind
// <nanobind/nanobind.h> の後に <nanobind/intrusive/ref.h> を定義しないと
// type_caster が有効にならないため、clang-format を無効化している。
// clang-format off
#include <nanobind/nanobind.h>
// clang-format on
#include <nanobind/intrusive/counter.h>
#include <nanobind/intrusive/ref.h>
#include <nanobind/make_iterator.h>
#include <nanobind/ndarray.h>
#include <nanobind/operators.h>
#include <nanobind/stl/chrono.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>
#include <nanobind/trampoline.h>
#include <nanobind/intrusive/counter.inl>

// libdatachannel
#include <rtc/rtc.hpp>

namespace nb = nanobind;
using namespace nb::literals;
using namespace rtc;

namespace nanobind {
namespace detail {

template <>
struct type_caster<std::vector<std::byte>> {
  NB_TYPE_CASTER(std::vector<std::byte>, const_name("bytes"));

  bool from_python(handle src, uint8_t flags, cleanup_list* cleanup) {
    if (PyBytes_Check(src.ptr()) == 0) {
      PyErr_Clear();
      return false;
    }
    size_t len = PyBytes_Size(src.ptr());
    const char* data = PyBytes_AsString(src.ptr());
    value.assign(reinterpret_cast<const std::byte*>(data),
                 reinterpret_cast<const std::byte*>(data + len));
    return true;
  }

  static handle from_cpp(const std::vector<std::byte>& vec,
                         rv_policy policy,
                         cleanup_list* cleanup) {
    return PyBytes_FromStringAndSize(reinterpret_cast<const char*>(vec.data()),
                                     vec.size());
  }
};
}  // namespace detail
}  // namespace nanobind

namespace {

// ---- configuration.hpp ----

void bind_configuration(nb::module_& m) {
  // IceServer
  nb::class_<IceServer> ice_server(m, "IceServer");

  nb::enum_<IceServer::Type>(ice_server, "Type")
      .value("Stun", IceServer::Type::Stun)
      .value("Turn", IceServer::Type::Turn);

  nb::enum_<IceServer::RelayType>(ice_server, "RelayType")
      .value("TurnUdp", IceServer::RelayType::TurnUdp)
      .value("TurnTcp", IceServer::RelayType::TurnTcp)
      .value("TurnTls", IceServer::RelayType::TurnTls);

  ice_server.def(nb::init<const std::string&>())
      .def(nb::init<std::string, uint16_t>())
      .def(nb::init<std::string, std::string>())
      .def(nb::init<std::string, uint16_t, std::string, std::string,
                    IceServer::RelayType>(),
           "hostname"_a, "port"_a, "username"_a, "password"_a,
           "relay_type"_a = IceServer::RelayType::TurnUdp)
      .def(nb::init<std::string, std::string, std::string, std::string,
                    IceServer::RelayType>(),
           "hostname"_a, "service"_a, "username"_a, "password"_a,
           "relay_type"_a = IceServer::RelayType::TurnUdp)
      .def_rw("hostname", &IceServer::hostname)
      .def_rw("port", &IceServer::port)
      .def_rw("type", &IceServer::type)
      .def_rw("username", &IceServer::username)
      .def_rw("password", &IceServer::password)
      .def_rw("relay_type", &IceServer::relayType);

  // ProxyServer
  nb::class_<ProxyServer> proxy_server(m, "ProxyServer");

  nb::enum_<ProxyServer::Type>(proxy_server, "Type")
      .value("Http", ProxyServer::Type::Http)
      .value("Socks5", ProxyServer::Type::Socks5);

  proxy_server.def(nb::init<const std::string&>())
      .def(nb::init<ProxyServer::Type, std::string, uint16_t>())
      .def(nb::init<ProxyServer::Type, std::string, uint16_t, std::string,
                    std::string>())
      .def_rw("type", &ProxyServer::type)
      .def_rw("hostname", &ProxyServer::hostname)
      .def_rw("port", &ProxyServer::port)
      .def_rw("username", &ProxyServer::username)
      .def_rw("password", &ProxyServer::password);

  // CertificateType
  nb::enum_<CertificateType>(m, "CertificateType")
      .value("Default", CertificateType::Default)
      .value("Ecdsa", CertificateType::Ecdsa)
      .value("Rsa", CertificateType::Rsa);

  // TransportPolicy
  nb::enum_<TransportPolicy>(m, "TransportPolicy")
      .value("All", TransportPolicy::All)
      .value("Relay", TransportPolicy::Relay);

  // Configuration
  nb::class_<Configuration>(m, "Configuration")
      .def(nb::init<>())
      .def_rw("ice_servers", &Configuration::iceServers)
      .def_rw("proxy_server", &Configuration::proxyServer)
      .def_rw("bind_address", &Configuration::bindAddress)
      .def_rw("certificate_type", &Configuration::certificateType)
      .def_rw("ice_transport_policy", &Configuration::iceTransportPolicy)
      .def_rw("enable_ice_tcp", &Configuration::enableIceTcp)
      .def_rw("enable_ice_udp_mux", &Configuration::enableIceUdpMux)
      .def_rw("disable_auto_negotiation",
              &Configuration::disableAutoNegotiation)
      .def_rw("disable_auto_gathering", &Configuration::disableAutoGathering)
      .def_rw("force_media_transport", &Configuration::forceMediaTransport)
      .def_rw("disable_fingerprint_verification",
              &Configuration::disableFingerprintVerification)
      .def_rw("port_range_begin", &Configuration::portRangeBegin)
      .def_rw("port_range_end", &Configuration::portRangeEnd)
      .def_rw("mtu", &Configuration::mtu)
      .def_rw("max_message_size", &Configuration::maxMessageSize)
      .def_rw("certificate_pem_file", &Configuration::certificatePemFile)
      .def_rw("key_pem_file", &Configuration::keyPemFile)
      .def_rw("key_pem_pass", &Configuration::keyPemPass);

  // WebSocketConfiguration
  nb::class_<WebSocketConfiguration>(m, "WebSocketConfiguration")
      .def(nb::init<>())
      .def_rw("disable_tls_verification",
              &WebSocketConfiguration::disableTlsVerification)
      .def_rw("proxy_server", &WebSocketConfiguration::proxyServer)
      .def_rw("protocols", &WebSocketConfiguration::protocols)
      .def_rw("connection_timeout", &WebSocketConfiguration::connectionTimeout)
      .def_rw("ping_interval", &WebSocketConfiguration::pingInterval)
      .def_rw("max_outstanding_pings",
              &WebSocketConfiguration::maxOutstandingPings)
      .def_rw("ca_certificate_pem_file",
              &WebSocketConfiguration::caCertificatePemFile)
      .def_rw("certificate_pem_file",
              &WebSocketConfiguration::certificatePemFile)
      .def_rw("key_pem_file", &WebSocketConfiguration::keyPemFile)
      .def_rw("key_pem_pass", &WebSocketConfiguration::keyPemPass)
      .def_rw("max_message_size", &WebSocketConfiguration::maxMessageSize);

  // WebSocketServerConfiguration
  nb::class_<WebSocketServerConfiguration>(m, "WebSocketServerConfiguration")
      .def(nb::init<>())
      .def_rw("port", &WebSocketServerConfiguration::port)
      .def_rw("enable_tls", &WebSocketServerConfiguration::enableTls)
      .def_rw("certificate_pem_file",
              &WebSocketServerConfiguration::certificatePemFile)
      .def_rw("key_pem_file", &WebSocketServerConfiguration::keyPemFile)
      .def_rw("key_pem_pass", &WebSocketServerConfiguration::keyPemPass)
      .def_rw("bind_address", &WebSocketServerConfiguration::bindAddress)
      .def_rw("connection_timeout",
              &WebSocketServerConfiguration::connectionTimeout)
      .def_rw("max_message_size",
              &WebSocketServerConfiguration::maxMessageSize);
}

// ---- description.hpp ----

nb::object get_media(Description& desc, int index) {
  auto var = desc.media(index);
  if (std::holds_alternative<Description::Media*>(var)) {
    return nb::cast(std::get<Description::Media*>(var));
  } else if (std::holds_alternative<Description::Application*>(var)) {
    return nb::cast(std::get<Description::Application*>(var));
  }
  return nb::none();
}

void bind_description(nb::module_& m) {
  // --- CertificateFingerprint ---
  nb::class_<CertificateFingerprint> cert_fp(m, "CertificateFingerprint");
  nb::enum_<CertificateFingerprint::Algorithm>(cert_fp, "Algorithm")
      .value("Sha1", CertificateFingerprint::Algorithm::Sha1)
      .value("Sha224", CertificateFingerprint::Algorithm::Sha224)
      .value("Sha256", CertificateFingerprint::Algorithm::Sha256)
      .value("Sha384", CertificateFingerprint::Algorithm::Sha384)
      .value("Sha512", CertificateFingerprint::Algorithm::Sha512);
  cert_fp.def(nb::init<>())
      .def_rw("algorithm", &CertificateFingerprint::algorithm)
      .def_rw("value", &CertificateFingerprint::value)
      .def("is_valid", &CertificateFingerprint::isValid)
      .def_static("algorithm_identifier",
                  &CertificateFingerprint::AlgorithmIdentifier)
      .def_static("algorithm_size", &CertificateFingerprint::AlgorithmSize);

  // --- Description::Type / Role / Direction ---
  nb::class_<Description> desc(m, "Description");

  nb::enum_<Description::Type>(desc, "Type")
      .value("Unspec", Description::Type::Unspec)
      .value("Offer", Description::Type::Offer)
      .value("Answer", Description::Type::Answer)
      .value("Pranswer", Description::Type::Pranswer)
      .value("Rollback", Description::Type::Rollback);

  nb::enum_<Description::Role>(desc, "Role")
      .value("ActPass", Description::Role::ActPass)
      .value("Passive", Description::Role::Passive)
      .value("Active", Description::Role::Active);

  nb::enum_<Description::Direction>(desc, "Direction")
      .value("SendOnly", Description::Direction::SendOnly)
      .value("RecvOnly", Description::Direction::RecvOnly)
      .value("SendRecv", Description::Direction::SendRecv)
      .value("Inactive", Description::Direction::Inactive)
      .value("Unknown", Description::Direction::Unknown);

  // Entry（抽象基底）
  nb::class_<Description::Entry> entry(desc, "Entry");

  // Entry::ExtMap
  nb::class_<Description::Entry::ExtMap> extmap(entry, "ExtMap");
  extmap
      .def(nb::init<int, std::string, Description::Direction>(), "id"_a,
           "uri"_a, "direction"_a = Description::Direction::Unknown)
      .def(nb::init<std::string_view>(), "description"_a)
      .def("set_description", &Description::Entry::ExtMap::setDescription)
      .def_static("parse_id", &Description::Entry::ExtMap::parseId)
      .def_rw("id", &Description::Entry::ExtMap::id)
      .def_rw("uri", &Description::Entry::ExtMap::uri)
      .def_rw("attributes", &Description::Entry::ExtMap::attributes)
      .def_rw("direction", &Description::Entry::ExtMap::direction);

  // Entry の本体
  entry.def("type", &Description::Entry::type)
      .def("protocol", &Description::Entry::protocol)
      .def("description", &Description::Entry::description)
      .def("mid", &Description::Entry::mid)
      .def("direction", &Description::Entry::direction)
      .def("set_direction", &Description::Entry::setDirection)
      .def("is_removed", &Description::Entry::isRemoved)
      .def("mark_removed", &Description::Entry::markRemoved)
      .def("attributes", &Description::Entry::attributes)
      .def("add_attribute", &Description::Entry::addAttribute)
      .def("remove_attribute", &Description::Entry::removeAttribute)
      .def("add_rid", &Description::Entry::addRid)
      .def("ext_ids", &Description::Entry::extIds)
      .def("add_ext_map", &Description::Entry::addExtMap)
      .def("remove_ext_map", &Description::Entry::removeExtMap)
      .def("__str__", [](const Description::Entry& e) {
        return static_cast<std::string>(e);
      });

  // Application（Entry を継承）
  nb::class_<Description::Application, Description::Entry>(desc, "Application")
      .def(nb::init<std::string>(), "mid"_a = "data")
      .def(nb::init<const std::string&, std::string>(), "mline"_a, "mid"_a)
      .def("reciprocate", &Description::Application::reciprocate)
      .def("set_sctp_port", &Description::Application::setSctpPort)
      .def("hint_sctp_port", &Description::Application::hintSctpPort)
      .def("set_max_message_size", &Description::Application::setMaxMessageSize)
      .def("sctp_port", &Description::Application::sctpPort)
      .def("max_message_size", &Description::Application::maxMessageSize)
      .def("parse_sdp_line", &Description::Application::parseSdpLine);

  // Media（Entry を継承）
  nb::class_<Description::Media, Description::Entry>(desc, "Media")
      .def(nb::init<const std::string&, std::string, Description::Direction>(),
           "mline"_a, "mid"_a, "direction"_a = Description::Direction::SendOnly)
      .def(nb::init<const std::string&>(), "sdp"_a)
      .def("description", &Description::Media::description)
      .def("reciprocate", &Description::Media::reciprocate)
      .def("add_ssrc", &Description::Media::addSSRC, "ssrc"_a, "name"_a,
           "msid"_a = std::nullopt, "track_id"_a = std::nullopt)
      .def("remove_ssrc", &Description::Media::removeSSRC)
      .def("replace_ssrc", &Description::Media::replaceSSRC)
      .def("has_ssrc", &Description::Media::hasSSRC)
      .def("clear_ssrcs", &Description::Media::clearSSRCs)
      .def("get_ssrcs", &Description::Media::getSSRCs)
      .def("get_cname_for_ssrc", &Description::Media::getCNameForSsrc)
      .def("bitrate", &Description::Media::bitrate)
      .def("set_bitrate", &Description::Media::setBitrate)
      .def("parse_sdp_line", &Description::Media::parseSdpLine)
      .def("has_payload_type", &Description::Media::hasPayloadType)
      .def("payload_types", &Description::Media::payloadTypes)
      .def("rtp_map", nb::overload_cast<int>(&Description::Media::rtpMap),
           "payload_type"_a, nb::rv_policy::reference)
      .def("add_rtp_map", &Description::Media::addRtpMap, "map"_a)
      .def("remove_rtp_map", &Description::Media::removeRtpMap,
           "payload_type"_a)
      .def("remove_format", &Description::Media::removeFormat, "format"_a)
      .def("add_rtx_codec", &Description::Media::addRtxCodec, "payload_type"_a,
           "orig_payload_type"_a, "clock_rate"_a)
      .def("as_audio",
           [](Description::Media* p) {
             return *static_cast<Description::Audio*>(p);
           })
      .def("as_video", [](Description::Media* p) {
        return *static_cast<Description::Video*>(p);
      });

  // RtpMap
  nb::class_<Description::Media::RtpMap> rtpmap(desc, "RtpMap");
  rtpmap.def(nb::init<int>(), "payload_type"_a)
      .def(nb::init<std::string_view>(), "description"_a)
      .def("set_description", &Description::Media::RtpMap::setDescription)
      .def_static("parse_payload_type",
                  &Description::Media::RtpMap::parsePayloadType)
      .def("add_feedback", &Description::Media::RtpMap::addFeedback)
      .def("remove_feedback", &Description::Media::RtpMap::removeFeedback)
      .def("add_parameter", &Description::Media::RtpMap::addParameter)
      .def("remove_parameter", &Description::Media::RtpMap::removeParameter)
      .def_rw("payload_type", &Description::Media::RtpMap::payloadType)
      .def_rw("format", &Description::Media::RtpMap::format)
      .def_rw("clock_rate", &Description::Media::RtpMap::clockRate)
      .def_rw("enc_params", &Description::Media::RtpMap::encParams)
      .def_rw("rtcp_fbs", &Description::Media::RtpMap::rtcpFbs)
      .def_rw("fmtps", &Description::Media::RtpMap::fmtps);

  // Media 継承: Audio
  nb::class_<Description::Audio, Description::Media>(desc, "Audio")
      .def(nb::init<std::string, Description::Direction>(), "mid"_a = "audio",
           "direction"_a = Description::Direction::SendOnly)
      .def("add_audio_codec", &Description::Audio::addAudioCodec,
           "payload_type"_a, "codec"_a, "profile"_a = std::nullopt)
      .def("add_opus_codec", &Description::Audio::addOpusCodec,
           "payload_type"_a, "profile"_a = DEFAULT_OPUS_AUDIO_PROFILE)
      .def("add_pcmu_codec", &Description::Audio::addPCMUCodec,
           "payload_type"_a, "profile"_a = std::nullopt)
      .def("add_pcmA_codec", &Description::Audio::addPCMACodec,
           "payload_type"_a, "profile"_a = std::nullopt)
      .def("add_aac_codec", &Description::Audio::addAACCodec, "payload_type"_a,
           "profile"_a = std::nullopt);

  // Media 継承: Video
  nb::class_<Description::Video, Description::Media>(desc, "Video")
      .def(nb::init<std::string, Description::Direction>(), "mid"_a = "video",
           "direction"_a = Description::Direction::SendOnly)
      .def("add_video_codec", &Description::Video::addVideoCodec,
           "payload_type"_a, "codec"_a, "profile"_a = std::nullopt)
      .def("add_h264_codec", &Description::Video::addH264Codec,
           "payload_type"_a, "profile"_a = DEFAULT_H264_VIDEO_PROFILE)
      .def("add_h265_codec", &Description::Video::addH265Codec,
           "payload_type"_a, "profile"_a = std::nullopt)
      .def("add_vp8_codec", &Description::Video::addVP8Codec, "payload_type"_a,
           "profile"_a = std::nullopt)
      .def("add_vp9_codec", &Description::Video::addVP9Codec, "payload_type"_a,
           "profile"_a = std::nullopt)
      .def("add_av1_codec", &Description::Video::addAV1Codec, "payload_type"_a,
           "profile"_a = std::nullopt);

  desc.def(nb::init<const std::string&, Description::Type, Description::Role>(),
           "sdp"_a, "type"_a = Description::Type::Unspec,
           "role"_a = Description::Role::ActPass)
      .def(nb::init<const std::string&, std::string>(), "sdp"_a,
           "type_string"_a)
      .def("type", &Description::type)
      .def("type_string", &Description::typeString)
      .def("role", &Description::role)
      .def("bundle_mid", &Description::bundleMid)
      .def("ice_options", &Description::iceOptions)
      .def("ice_ufrag", &Description::iceUfrag)
      .def("ice_pwd", &Description::icePwd)
      .def("fingerprint", &Description::fingerprint)
      .def("hint_type", &Description::hintType)
      .def("set_fingerprint", &Description::setFingerprint)
      .def("add_ice_option", &Description::addIceOption)
      .def("remove_ice_option", &Description::removeIceOption)
      .def("__str__",
           [](const Description& d) { return static_cast<std::string>(d); })
      .def("has_application", &Description::hasApplication)
      .def("has_audio_or_video", &Description::hasAudioOrVideo)
      .def("has_mid", &Description::hasMid, "mid"_a)
      .def("add_media",
           nb::overload_cast<Description::Media>(&Description::addMedia),
           "media"_a)
      .def("add_media",
           nb::overload_cast<Description::Application>(&Description::addMedia),
           "application"_a)
      .def("add_application", &Description::addApplication, "mid"_a = "data")
      .def("add_video", &Description::addVideo, "mid"_a = "video",
           "dir"_a = Description::Direction::SendOnly)
      .def("add_audio", &Description::addAudio, "mid"_a = "audio",
           "dir"_a = Description::Direction::SendOnly)
      .def("clear_media", &Description::clearMedia)
      .def("media", &get_media)
      .def("media_count", &Description::mediaCount)
      .def(
          "application",
          [](Description& desc) -> Description::Application* {
            return desc.application();
          },
          nb::rv_policy::reference);
}

// ---- candidate.hpp ----

void bind_candidate(nb::module_& m) {
  // Candidate enums
  nb::class_<Candidate> candidate(m, "Candidate");

  nb::enum_<Candidate::Family>(candidate, "Family")
      .value("Unresolved", Candidate::Family::Unresolved)
      .value("Ipv4", Candidate::Family::Ipv4)
      .value("Ipv6", Candidate::Family::Ipv6);

  nb::enum_<Candidate::Type>(candidate, "Type")
      .value("Unknown", Candidate::Type::Unknown)
      .value("Host", Candidate::Type::Host)
      .value("ServerReflexive", Candidate::Type::ServerReflexive)
      .value("PeerReflexive", Candidate::Type::PeerReflexive)
      .value("Relayed", Candidate::Type::Relayed);

  nb::enum_<Candidate::TransportType>(candidate, "TransportType")
      .value("Unknown", Candidate::TransportType::Unknown)
      .value("Udp", Candidate::TransportType::Udp)
      .value("TcpActive", Candidate::TransportType::TcpActive)
      .value("TcpPassive", Candidate::TransportType::TcpPassive)
      .value("TcpSo", Candidate::TransportType::TcpSo)
      .value("TcpUnknown", Candidate::TransportType::TcpUnknown);

  nb::enum_<Candidate::ResolveMode>(candidate, "ResolveMode")
      .value("Simple", Candidate::ResolveMode::Simple)
      .value("Lookup", Candidate::ResolveMode::Lookup);

  // Candidate class
  candidate.def(nb::init<>())
      .def(nb::init<std::string>())
      .def(nb::init<std::string, std::string>(), "candidate"_a, "mid"_a)
      .def("hint_mid", &Candidate::hintMid)
      .def("change_address",
           nb::overload_cast<std::string>(&Candidate::changeAddress))
      .def("change_address",
           nb::overload_cast<std::string, uint16_t>(&Candidate::changeAddress))
      .def("change_address", nb::overload_cast<std::string, std::string>(
                                 &Candidate::changeAddress))
      .def("resolve", &Candidate::resolve,
           "mode"_a = Candidate::ResolveMode::Simple)
      .def("type", &Candidate::type)
      .def("transport_type", &Candidate::transportType)
      .def("priority", &Candidate::priority)
      .def("candidate", &Candidate::candidate)
      .def("mid", &Candidate::mid)
      .def("is_resolved", &Candidate::isResolved)
      .def("family", &Candidate::family)
      .def("address", &Candidate::address)
      .def("port", &Candidate::port)
      .def(nb::self == nb::self,
           nb::sig("def __eq__(self, arg: object, /) -> bool"))
      .def(nb::self != nb::self,
           nb::sig("def __ne__(self, arg: object, /) -> bool"))
      .def("__str__",
           [](const Candidate& c) { return static_cast<std::string>(c); });
}

// ---- reliability.hpp ----

void bind_reliability(nb::module_& m) {
  nb::class_<Reliability>(m, "Reliability")
      .def(nb::init<>())
      .def_rw("unordered", &Reliability::unordered)
      .def_rw("max_packet_lifetime", &Reliability::maxPacketLifeTime)
      .def_rw("max_retransmits", &Reliability::maxRetransmits);
}

// ---- frameinfo.hpp ----

void bind_frameinfo(nb::module_& m) {
  nb::class_<FrameInfo>(m, "FrameInfo")
      .def(nb::init<uint8_t, uint32_t>(), "payload_type"_a, "timestamp"_a)
      .def_rw("payload_type", &FrameInfo::payloadType)
      .def_rw("timestamp", &FrameInfo::timestamp);
}

// ---- message.hpp ----

void bind_message(nb::module_& m) {
  // Message::Type enum
  nb::class_<Message> message(m, "Message");

  nb::enum_<Message::Type>(message, "Type")
      .value("Binary", Message::Type::Binary)
      .value("String", Message::Type::String)
      .value("Control", Message::Type::Control)
      .value("Reset", Message::Type::Reset);

  // Message class
  message
      .def(nb::init<size_t, Message::Type>(), "size"_a,
           "type"_a = Message::Type::Binary)
      .def_prop_rw(
          "type", [](const Message& m) { return m.type; },
          [](Message& m, Message::Type t) { m.type = t; })
      .def_rw("stream", &Message::stream)
      .def_rw("dscp", &Message::dscp)
      .def_rw("reliability", &Message::reliability)
      .def_rw("frame_info", &Message::frameInfo)
      .def("to_bytes",
           [](const Message& m) -> nb::bytes {
             return nb::bytes(reinterpret_cast<const char*>(m.data()),
                              m.size());
           })
      .def("to_str",
           [](const Message& m) -> std::string {
             return std::string(reinterpret_cast<const char*>(m.data()),
                                m.size());
           })
      .def("__len__", [](const Message& m) { return m.size(); })
      .def("__getitem__",
           [](const Message& m, size_t i) -> int {
             if (i >= m.size())
               throw std::out_of_range("Message index out of range");
             return (int)m[i];
           })
      .def("__setitem__", [](Message& m, size_t i, int b) {
        if (i >= m.size())
          throw std::out_of_range("Message index out of range");
        m[i] = (std::byte)b;
      });

  // make_message overloads
  m.def(
      "make_message",
      [](size_t size, Message::Type type, unsigned int stream,
         std::shared_ptr<Reliability> reliability) {
        return make_message(size, type, stream, reliability);
      },
      "size"_a, "type"_a = Message::Binary, "stream"_a = 0,
      "reliability"_a = nullptr);

  m.def(
      "make_message_from_data",
      [](std::vector<byte> data, Message::Type type, unsigned int stream,
         std::shared_ptr<Reliability> reliability,
         std::shared_ptr<FrameInfo> frameInfo) {
        return make_message(std::move(data), type, stream, reliability,
                            frameInfo);
      },
      "data"_a, "type"_a = Message::Binary, "stream"_a = 0,
      "reliability"_a = nullptr, "frame_info"_a = nullptr);

  m.def("make_message_from_variant",
        static_cast<message_ptr (*)(message_variant)>(&make_message), "data"_a);

  m.def("message_size", &message_size_func, "message"_a);

  m.def(
      "to_variant",
      [](const Message& msg) -> nb::object {
        auto var = to_variant(msg);
        if (std::holds_alternative<std::string>(var)) {
          const auto& str = std::get<std::string>(var);
          return nb::str(str.c_str(), str.size());
        } else {
          const auto& vec = std::get<binary>(var);
          return nb::bytes(reinterpret_cast<const char*>(vec.data()),
                           vec.size());
        }
      },
      "message"_a);
}

// ---- nalunit.hpp ----

void bind_nalunit(nb::module_& m) {
  // --- NalUnitHeader ---
  nb::class_<NalUnitHeader>(m, "NalUnitHeader")
      .def(nb::init<>())
      .def("forbidden_bit", &NalUnitHeader::forbiddenBit)
      .def("nri", &NalUnitHeader::nri)
      .def("idc", &NalUnitHeader::idc)
      .def("unit_type", &NalUnitHeader::unitType)
      .def("set_forbidden_bit", &NalUnitHeader::setForbiddenBit)
      .def("set_nri", &NalUnitHeader::setNRI)
      .def("set_unit_type", &NalUnitHeader::setUnitType);

  // --- NalUnitFragmentHeader ---
  nb::class_<NalUnitFragmentHeader>(m, "NalUnitFragmentHeader")
      .def(nb::init<>())
      .def("is_start", &NalUnitFragmentHeader::isStart)
      .def("reserved_bit6", &NalUnitFragmentHeader::reservedBit6)
      .def("is_end", &NalUnitFragmentHeader::isEnd)
      .def("unit_type", &NalUnitFragmentHeader::unitType)
      .def("set_start", &NalUnitFragmentHeader::setStart)
      .def("set_end", &NalUnitFragmentHeader::setEnd)
      .def("set_reserved_bit6", &NalUnitFragmentHeader::setReservedBit6)
      .def("set_unit_type", &NalUnitFragmentHeader::setUnitType);

  // --- NalUnitStartSequenceMatch enum ---
  nb::enum_<NalUnitStartSequenceMatch>(m, "NalUnitStartSequenceMatch")
      .value("NoMatch", NUSM_noMatch)
      .value("FirstZero", NUSM_firstZero)
      .value("SecondZero", NUSM_secondZero)
      .value("ThirdZero", NUSM_thirdZero)
      .value("ShortMatch", NUSM_shortMatch)
      .value("LongMatch", NUSM_longMatch);

  // --- NalUnit::Separator ---
  nb::class_<NalUnit> nalunit(m, "NalUnit");

  nb::enum_<NalUnit::Type>(nalunit, "Type")
      .value("H264", NalUnit::Type::H264)
      .value("H265", NalUnit::Type::H265);

  nb::enum_<NalUnit::Separator>(nalunit, "Separator")
      .value("Length", NalUnit::Separator::Length)
      .value("LongStartSequence", NalUnit::Separator::LongStartSequence)
      .value("ShortStartSequence", NalUnit::Separator::ShortStartSequence)
      .value("StartSequence", NalUnit::Separator::StartSequence);

  nalunit.def(nb::init<>())
      .def(nb::init<size_t, bool, NalUnit::Type>(), "size"_a,
           "including_header"_a = true, "type"_a = NalUnit::Type::H264)
      .def(nb::init<binary&&>())
      .def("forbidden_bit", &NalUnit::forbiddenBit)
      .def("nri", &NalUnit::nri)
      .def("unit_type", &NalUnit::unitType)
      .def("payload", &NalUnit::payload)
      .def("set_forbidden_bit", &NalUnit::setForbiddenBit)
      .def("set_nri", &NalUnit::setNRI)
      .def("set_unit_type", &NalUnit::setUnitType)
      .def("set_payload", &NalUnit::setPayload)
      .def_static(
          "start_sequence_match_succ",
          [](NalUnitStartSequenceMatch match, int _byte,
             NalUnit::Separator separator) {
            return NalUnit::StartSequenceMatchSucc(match, (std::byte)_byte,
                                                   separator);
          },
          "match"_a, "_byte"_a, "separator"_a);

  nb::class_<NalUnitFragmentA, NalUnit> fragment(m, "NalUnitFragmentA");

  // --- NalUnitFragmentA::FragmentType ---
  nb::enum_<NalUnitFragmentA::FragmentType>(fragment, "FragmentType")
      .value("Start", NalUnitFragmentA::FragmentType::Start)
      .value("Middle", NalUnitFragmentA::FragmentType::Middle)
      .value("End", NalUnitFragmentA::FragmentType::End);

  // --- NalUnitFragmentA ---
  fragment
      .def(nb::init<NalUnitFragmentA::FragmentType, bool, uint8_t, uint8_t,
                    binary>(),
           "type"_a, "forbidden_bit"_a, "nri"_a, "unit_type"_a, "data"_a)
      .def("unit_type", &NalUnitFragmentA::unitType)
      .def("payload", &NalUnitFragmentA::payload)
      .def("type", &NalUnitFragmentA::type)
      .def("set_unit_type", &NalUnitFragmentA::setUnitType)
      .def("set_payload", &NalUnitFragmentA::setPayload)
      .def("set_fragment_type", &NalUnitFragmentA::setFragmentType)
      .def_static("fragments_from", &NalUnitFragmentA::fragmentsFrom, "nalu"_a,
                  "max_fragment_size"_a);

  // --- NalUnits helper class ---
  nb::class_<NalUnits>(m, "NalUnits")
      .def(nb::init<>())
      .def(
          "generate_fragments",
          [](NalUnits& v, uint16_t max_fragment_size) {
            std::vector<std::shared_ptr<binary>> xs =
                v.generateFragments(max_fragment_size);
            std::vector<binary> result;
            for (const auto& x : xs) {
              result.push_back(*x);
            }
            return result;
          },
          "max_fragment_size"_a)
      .def_prop_ro_static("DEFAULT_MAXIMUM_FRAGMENT_SIZE", [](nb::handle) {
        return NalUnits::defaultMaximumFragmentSize;
      });
}

const uint16_t NalUnits_defaultMaximumFragmentSize =
    NalUnits::defaultMaximumFragmentSize;

// ---- h265nalunit.hpp ----

void bind_h265nalunit(nb::module_& m) {
  // --- H265NalUnitHeader ---
  nb::class_<H265NalUnitHeader>(m, "H265NalUnitHeader")
      .def(nb::init<>())
      .def("forbidden_bit", &H265NalUnitHeader::forbiddenBit)
      .def("unit_type", &H265NalUnitHeader::unitType)
      .def("nuh_layer_id", &H265NalUnitHeader::nuhLayerId)
      .def("nuh_temp_id_plus1", &H265NalUnitHeader::nuhTempIdPlus1)
      .def("set_forbidden_bit", &H265NalUnitHeader::setForbiddenBit)
      .def("set_unit_type", &H265NalUnitHeader::setUnitType)
      .def("set_nuh_layer_id", &H265NalUnitHeader::setNuhLayerId)
      .def("set_nuh_temp_id_plus1", &H265NalUnitHeader::setNuhTempIdPlus1);

  // --- H265NalUnitFragmentHeader ---
  nb::class_<H265NalUnitFragmentHeader>(m, "H265NalUnitFragmentHeader")
      .def(nb::init<>())
      .def("is_start", &H265NalUnitFragmentHeader::isStart)
      .def("is_end", &H265NalUnitFragmentHeader::isEnd)
      .def("unit_type", &H265NalUnitFragmentHeader::unitType)
      .def("set_start", &H265NalUnitFragmentHeader::setStart)
      .def("set_end", &H265NalUnitFragmentHeader::setEnd)
      .def("set_unit_type", &H265NalUnitFragmentHeader::setUnitType);

  // --- H265NalUnit ---
  nb::class_<H265NalUnit, NalUnit>(m, "H265NalUnit")
      .def(nb::init<>())
      .def(nb::init<size_t, bool>(), "size"_a, "including_header"_a = true)
      .def(nb::init<binary&&>(), "data"_a)
      .def("forbidden_bit", &H265NalUnit::forbiddenBit)
      .def("unit_type", &H265NalUnit::unitType)
      .def("nuh_layer_id", &H265NalUnit::nuhLayerId)
      .def("nuh_temp_id_plus1", &H265NalUnit::nuhTempIdPlus1)
      .def("payload", &H265NalUnit::payload)
      .def("set_forbidden_bit", &H265NalUnit::setForbiddenBit)
      .def("set_unit_type", &H265NalUnit::setUnitType)
      .def("set_nuh_layer_id", &H265NalUnit::setNuhLayerId)
      .def("set_nuh_temp_id_plus1", &H265NalUnit::setNuhTempIdPlus1)
      .def("set_payload", &H265NalUnit::setPayload);

  nb::class_<H265NalUnitFragment, H265NalUnit> fragment(m,
                                                        "H265NalUnitFragment");

  // --- H265NalUnitFragment::FragmentType ---
  nb::enum_<H265NalUnitFragment::FragmentType>(fragment, "FragmentType")
      .value("Start", H265NalUnitFragment::FragmentType::Start)
      .value("Middle", H265NalUnitFragment::FragmentType::Middle)
      .value("End", H265NalUnitFragment::FragmentType::End);

  // --- H265NalUnitFragment ---
  fragment
      .def(nb::init<H265NalUnitFragment::FragmentType, bool, uint8_t, uint8_t,
                    uint8_t, binary>(),
           "type"_a, "forbidden_bit"_a, "nuh_layer_id"_a, "nuh_temp_id_plus1"_a,
           "unit_type"_a, "data"_a)
      .def("unit_type", &H265NalUnitFragment::unitType)
      .def("payload", &H265NalUnitFragment::payload)
      .def("type", &H265NalUnitFragment::type)
      .def("set_unit_type", &H265NalUnitFragment::setUnitType)
      .def("set_payload", &H265NalUnitFragment::setPayload)
      .def("set_fragment_type", &H265NalUnitFragment::setFragmentType)
      .def_static("fragments_from", &H265NalUnitFragment::fragmentsFrom,
                  "nalu"_a, "max_fragment_size"_a);

  // --- H265NalUnits ---
  nb::class_<H265NalUnits>(m, "H265NalUnits")
      .def(nb::init<>())
      .def(
          "generate_fragments",
          [](H265NalUnits& v, uint16_t max_fragment_size) {
            std::vector<std::shared_ptr<binary>> xs =
                v.generateFragments(max_fragment_size);
            std::vector<binary> result;
            for (const auto& x : xs) {
              result.push_back(*x);
            }
            return result;
          },
          "max_fragment_size"_a)
      .def_prop_ro_static("DEFAULT_MAXIMUM_FRAGMENT_SIZE", [](nb::handle) {
        return H265NalUnits::defaultMaximumFragmentSize;
      });
}

const uint16_t H265NalUnits_defaultMaximumFragmentSize =
    H265NalUnits::defaultMaximumFragmentSize;

// ---- mediahandler.hpp ----

class PyMediaHandler : public MediaHandler {};
class PyMediaHandlerImpl : public PyMediaHandler {
 public:
  NB_TRAMPOLINE(PyMediaHandler, 5);
  void media(const Description::Media& desc) override {
    NB_OVERRIDE(media, desc);
  }
  void incoming(message_vector& messages,
                const message_callback& send) override {
    NB_OVERRIDE(incoming, messages, send);
  }
  void outgoing(message_vector& messages,
                const message_callback& send) override {
    NB_OVERRIDE(outgoing, messages, send);
  }
  bool requestKeyframe(const message_callback& send) override {
    NB_OVERRIDE_NAME("request_keyframe", requestKeyframe, send);
  }
  bool requestBitrate(unsigned int bitrate,
                      const message_callback& send) override {
    NB_OVERRIDE_NAME("request_bitrate", requestBitrate, bitrate, send);
  }
};

void bind_mediahandler(nb::module_& m) {
  nb::class_<MediaHandler>(m, "MediaHandler")
      .def(nb::init<>())
      .def(
          "media",
          [](std::shared_ptr<MediaHandler> self,
             const Description::Media& desc) { self->media(desc); },
          "desc"_a)
      .def(
          "incoming",
          [](std::shared_ptr<MediaHandler> self, nb::list py_messages,
             const message_callback& send) -> nb::list {
            // Python リストを C++ ベクトルに変換
            message_vector messages;
            for (auto item : py_messages) {
              messages.push_back(nb::cast<message_ptr>(item));
            }
            
            // C++ の incoming を呼び出す
            self->incoming(messages, send);
            
            // 変更後の C++ ベクトルを Python リストとして返す
            nb::list result;
            for (auto& msg : messages) {
              result.append(msg);
            }
            return result;
          },
          "messages"_a, "send"_a)
      .def(
          "outgoing",
          [](std::shared_ptr<MediaHandler> self, nb::list py_messages,
             const message_callback& send) -> nb::list {
            // Python リストを C++ ベクトルに変換
            message_vector messages;
            for (auto item : py_messages) {
              messages.push_back(nb::cast<message_ptr>(item));
            }
            
            // C++ の outgoing を呼び出す
            self->outgoing(messages, send);
            
            // 変更後の C++ ベクトルを Python リストとして返す
            nb::list result;
            for (auto& msg : messages) {
              result.append(msg);
            }
            return result;
          },
          "messages"_a, "send"_a)
      .def(
          "request_keyframe",
          [](std::shared_ptr<MediaHandler> self, const message_callback& send) {
            return self->requestKeyframe(send);
          },
          "send"_a)
      .def(
          "request_bitrate",
          [](std::shared_ptr<MediaHandler> self, unsigned int bitrate,
             const message_callback& send) {
            return self->requestBitrate(bitrate, send);
          },
          "bitrate"_a, "send"_a)
      .def(
          "add_to_chain",
          [](std::shared_ptr<MediaHandler> self,
             std::shared_ptr<MediaHandler> handler) {
            self->addToChain(handler);
          },
          "handler"_a)
      .def("set_next", &MediaHandler::setNext, "next"_a)
      .def("next",
           [](std::shared_ptr<MediaHandler> self) { return self->next(); })
      .def("last",
           [](std::shared_ptr<MediaHandler> self) { return self->last(); })
      .def(
          "media_chain",
          [](std::shared_ptr<MediaHandler> self,
             const Description::Media& desc) { return self->mediaChain(desc); },
          "desc"_a)
      .def(
          "incoming_chain",
          [](std::shared_ptr<MediaHandler> self, message_vector& messages,
             const message_callback& send) {
            return self->incomingChain(messages, send);
          },
          "messages"_a, "send"_a)
      .def(
          "outgoing_chain",
          [](std::shared_ptr<MediaHandler> self, message_vector& messages,
             const message_callback& send) {
            return self->outgoingChain(messages, send);
          },
          "messages"_a, "send"_a);

  nb::class_<PyMediaHandler, PyMediaHandlerImpl, MediaHandler>(m,
                                                               "PyMediaHandler")
      .def(nb::init<>());
}

// ---- rtppacketizationconfig.hpp ----

void bind_rtppacketizationconfig(nb::module_& m) {
  nb::class_<RtpPacketizationConfig>(m, "RtpPacketizationConfig")
      .def(nb::init<uint32_t, std::string, uint8_t, uint32_t, uint8_t>(),
           "ssrc"_a, "cname"_a, "payload_type"_a, "clock_rate"_a,
           "video_orientation_id"_a = 0)

      // Fields
      .def_rw("ssrc", &RtpPacketizationConfig::ssrc)
      .def_rw("cname", &RtpPacketizationConfig::cname)
      .def_rw("payload_type", &RtpPacketizationConfig::payloadType)
      .def_rw("clock_rate", &RtpPacketizationConfig::clockRate)
      .def_rw("video_orientation_id",
              &RtpPacketizationConfig::videoOrientationId)
      .def_rw("sequence_number", &RtpPacketizationConfig::sequenceNumber)
      .def_rw("timestamp", &RtpPacketizationConfig::timestamp)
      .def_rw("start_timestamp", &RtpPacketizationConfig::startTimestamp)
      .def_rw("video_orientation", &RtpPacketizationConfig::videoOrientation)
      .def_rw("mid_id", &RtpPacketizationConfig::midId)
      .def_rw("mid", &RtpPacketizationConfig::mid)
      .def_rw("rid_id", &RtpPacketizationConfig::ridId)
      .def_rw("rid", &RtpPacketizationConfig::rid)
      .def_rw("playout_delay_id", &RtpPacketizationConfig::playoutDelayId)
      .def_rw("playout_delay_min", &RtpPacketizationConfig::playoutDelayMin)
      .def_rw("playout_delay_max", &RtpPacketizationConfig::playoutDelayMax)

      // Methods
      .def_static("get_seconds_from_timestamp",
                  &RtpPacketizationConfig::getSecondsFromTimestamp,
                  "timestamp"_a, "clock_rate"_a)

      .def("timestamp_to_seconds", &RtpPacketizationConfig::timestampToSeconds,
           "timestamp"_a)

      .def_static("get_timestamp_from_seconds",
                  &RtpPacketizationConfig::getTimestampFromSeconds, "seconds"_a,
                  "clock_rate"_a)

      .def("seconds_to_timestamp", &RtpPacketizationConfig::secondsToTimestamp,
           "seconds"_a);
}

// ---- rtppacketizer.hpp ----

void bind_rtppacketizer(nb::module_& m) {
  nb::class_<RtpPacketizer, MediaHandler>(m, "RtpPacketizer")
      .def(nb::init<std::shared_ptr<RtpPacketizationConfig>>(), "rtp_config"_a)
      .def("media", &RtpPacketizer::media)
      .def("outgoing", &RtpPacketizer::outgoing)
      .def_prop_ro("rtp_config",
                   [](const RtpPacketizer& self) { return self.rtpConfig; });

  nb::class_<OpusRtpPacketizer, RtpPacketizer>(m, "OpusRtpPacketizer")
      .def(nb::init<std::shared_ptr<RtpPacketizationConfig>>(), "rtp_config"_a);

  // OpusRtpPacketizer と AACRtpPacketizer は同じ型なので片方しか登録できないため
  // AACRtpPacketizer は __init__.py で別名を作る。
  static_assert(std::is_same_v<OpusRtpPacketizer, AACRtpPacketizer>,
                "OpusRtpPacketizer and AACRtpPacketizer should be the same");
  // nb::class_<AACRtpPacketizer, RtpPacketizer>(m, "AACRtpPacketizer")
  //     .def(nb::init<std::shared_ptr<RtpPacketizationConfig>>(), "rtp_config"_a);
}

// ---- av1rtppacketizer.hpp ----

void bind_av1rtppacketizer(nb::module_& m) {
  nb::class_<AV1RtpPacketizer, RtpPacketizer> av1pkt(m, "AV1RtpPacketizer");

  // Nested enum
  nb::enum_<AV1RtpPacketizer::Packetization>(av1pkt, "Packetization")
      .value("Obu", AV1RtpPacketizer::Packetization::Obu)
      .value("TemporalUnit", AV1RtpPacketizer::Packetization::TemporalUnit);

  av1pkt
      .def(nb::init<AV1RtpPacketizer::Packetization,
                    std::shared_ptr<RtpPacketizationConfig>, uint16_t>(),
           "packetization"_a, "rtp_config"_a,
           "max_fragment_size"_a = NalUnits_defaultMaximumFragmentSize)
      .def("outgoing", &AV1RtpPacketizer::outgoing)
      .def_prop_ro_static("DEFAULT_CLOCK_RATE", [](nb::handle) {
        return AV1RtpPacketizer::defaultClockRate;
      });
}

// ---- h264rtppacketizer.hpp ----

void bind_h264rtppacketizer(nb::module_& m) {
  nb::class_<H264RtpPacketizer, RtpPacketizer>(m, "H264RtpPacketizer")
      .def(nb::init<NalUnit::Separator, std::shared_ptr<RtpPacketizationConfig>,
                    uint16_t>(),
           "separator"_a, "rtp_config"_a,
           "max_fragment_size"_a = NalUnits_defaultMaximumFragmentSize)
      .def("outgoing", &H264RtpPacketizer::outgoing)
      .def_prop_ro_static("DEFAULT_CLOCK_RATE", [](nb::handle) {
        return H264RtpPacketizer::defaultClockRate;
      });
}

// ---- h265rtppacketizer.hpp ----

void bind_h265rtppacketizer(nb::module_& m) {
  nb::class_<H265RtpPacketizer, RtpPacketizer>(m, "H265RtpPacketizer")
      .def(nb::init<NalUnit::Separator, std::shared_ptr<RtpPacketizationConfig>,
                    uint16_t>(),
           "separator"_a, "rtp_config"_a,
           "max_fragment_size"_a = H265NalUnits_defaultMaximumFragmentSize)
      .def("outgoing", &H265RtpPacketizer::outgoing)
      .def_prop_ro_static("DEFAULT_CLOCK_RATE", [](nb::handle) {
        return H265RtpPacketizer::defaultClockRate;
      });
}

// ---- rtpdepacketizer.hpp ----

void bind_rtpdepacketizer(nb::module_& m) {
  nb::class_<RtpDepacketizer, MediaHandler>(m, "RtpDepacketizer")
      .def(nb::init<>())
      .def(
          "incoming",
          [](std::shared_ptr<RtpDepacketizer> self, nb::list py_messages,
             const message_callback& send) -> nb::list {
            // Python リストを C++ ベクトルに変換
            message_vector messages;
            for (auto item : py_messages) {
              messages.push_back(nb::cast<message_ptr>(item));
            }
            
            // C++ の incoming を呼び出す（messages が in-place で変更される）
            self->incoming(messages, send);
            
            // 変更後の C++ ベクトルを Python リストとして返す
            nb::list result;
            for (auto& msg : messages) {
              result.append(msg);
            }
            return result;
          },
          "messages"_a, "send"_a,
          "Process incoming RTP messages and return depacketized messages");
}

// ---- h264depacketizer.hpp ----

void bind_h264depacketizer(nb::module_& m) {
  nb::class_<H264RtpDepacketizer, MediaHandler>(m, "H264RtpDepacketizer")
      .def(nb::init<NalUnit::Separator>(),
           "separator"_a = NalUnit::Separator::LongStartSequence)
      .def(
          "incoming",
          [](std::shared_ptr<H264RtpDepacketizer> self, nb::list py_messages,
             const message_callback& send) -> nb::list {
            // Python リストを C++ ベクトルに変換
            message_vector messages;
            for (auto item : py_messages) {
              messages.push_back(nb::cast<message_ptr>(item));
            }
            
            // C++ の incoming を呼び出す（messages が in-place で変更される）
            self->incoming(messages, send);
            
            // 変更後の C++ ベクトルを Python リストとして返す
            nb::list result;
            for (auto& msg : messages) {
              result.append(msg);
            }
            return result;
          },
          "messages"_a, "send"_a,
          "Process incoming RTP messages and return depacketized NAL units");
}

// ---- pacinghandler.hpp ----

void bind_pacinghandler(nb::module_& m) {
  nb::class_<PacingHandler, MediaHandler>(m, "PacingHandler")
      .def(nb::init<double, std::chrono::milliseconds>(), "bits_per_second"_a,
           "send_interval"_a)
      .def("outgoing", &PacingHandler::outgoing);
}

// ---- rembhandler.hpp ----

void bind_rembhandler(nb::module_& m) {
  nb::class_<RembHandler, MediaHandler>(m, "RembHandler")
      .def(nb::init<std::function<void(unsigned int)>>(), "on_remb"_a)
      .def("incoming", &RembHandler::incoming);
}

// ---- plihandler.hpp ----

void bind_plihandler(nb::module_& m) {
  nb::class_<PliHandler, MediaHandler>(m, "PliHandler")
      .def(nb::init<std::function<void()>>(), "on_pli"_a)
      .def("incoming", &PliHandler::incoming);
}

// ---- rtcpnackresponder.hpp ----

const size_t RtcpNackResponder_DefaultMaxSize =
    RtcpNackResponder::DefaultMaxSize;

void bind_rtcpnackresponder(nb::module_& m) {
  nb::class_<RtcpNackResponder, MediaHandler>(m, "RtcpNackResponder")
      .def(nb::init<size_t>(), "max_size"_a = RtcpNackResponder_DefaultMaxSize)
      .def_prop_ro_static(
          "DEFAULT_MAX_SIZE",
          [](nb::handle) { return RtcpNackResponder_DefaultMaxSize; })
      .def("incoming", &RtcpNackResponder::incoming)
      .def("outgoing", &RtcpNackResponder::outgoing);
}

// ---- rtcpreceivingsession.hpp ----

void bind_rtcpreceivingsession(nb::module_& m) {
  nb::class_<RtcpReceivingSession, MediaHandler>(m, "RtcpReceivingSession")
      .def(nb::init<>())
      .def("incoming", &RtcpReceivingSession::incoming)
      .def("request_keyframe", nb::overload_cast<const message_callback&>(
                                   &RtcpReceivingSession::requestKeyframe))
      .def("request_bitrate",
           nb::overload_cast<unsigned int, const message_callback&>(
               &RtcpReceivingSession::requestBitrate));
}

// ---- rtcpsrreporter.hpp ----

void bind_rtcpsrreporter(nb::module_& m) {
  nb::class_<RtcpSrReporter, MediaHandler>(m, "RtcpSrReporter")
      .def(nb::init<std::shared_ptr<RtpPacketizationConfig>>(), "rtp_config"_a)
      .def("last_reported_timestamp", &RtcpSrReporter::lastReportedTimestamp)
      .def("set_needs_to_report", &RtcpSrReporter::setNeedsToReport)
      .def("outgoing", &RtcpSrReporter::outgoing)
      .def_prop_ro("rtp_config",
                   [](const RtcpSrReporter& self) { return self.rtpConfig; });
}

// ---- channel.hpp ----

void bind_channel(nb::module_& m) {
  nb::class_<Channel>(m, "Channel")
      // Core API
      .def("close", &Channel::close)
      .def("send", nb::overload_cast<message_variant>(&Channel::send), "data"_a)
      .def(
          "send",
          [](Channel& self, std::vector<byte> data, size_t size) {
            return self.send(data.data(), size);
          },
          "data"_a, "size"_a)
      .def("is_open", &Channel::isOpen)
      .def("is_closed", &Channel::isClosed)
      .def("max_message_size", &Channel::maxMessageSize)
      .def("buffered_amount", &Channel::bufferedAmount)

      // Callback registration
      .def("on_open", &Channel::onOpen)
      .def("on_closed", &Channel::onClosed)
      .def("on_error", &Channel::onError)
      .def("on_message",
           nb::overload_cast<std::function<void(message_variant)>>(
               &Channel::onMessage))
      .def("on_message",
           nb::overload_cast<std::function<void(binary)>,
                             std::function<void(std::string)>>(
               &Channel::onMessage),
           "binary_callback"_a, "string_callback"_a)
      .def("on_buffered_amount_low", &Channel::onBufferedAmountLow)
      .def("set_buffered_amount_low_threshold",
           &Channel::setBufferedAmountLowThreshold)
      .def("reset_callbacks", &Channel::resetCallbacks)

      // Extended API
      .def("receive", &Channel::receive)
      .def("peek", &Channel::peek)
      .def("available_amount", &Channel::availableAmount)
      .def("on_available", &Channel::onAvailable);
}

// ---- datachannel.hpp ----

void bind_datachannel(nb::module_& m) {
  nb::class_<DataChannel, Channel>(m, "DataChannel")
      .def("is_open", &DataChannel::isOpen)
      .def("is_closed", &DataChannel::isClosed)
      .def("max_message_size", &DataChannel::maxMessageSize)
      .def("close", &DataChannel::close)
      .def("send", nb::overload_cast<message_variant>(&DataChannel::send),
           "data"_a)
      .def(
          "send",
          [](DataChannel& self, std::vector<byte> data, size_t size) {
            return self.send(data.data(), size);
          },
          "data"_a, "size"_a)
      .def("stream", &DataChannel::stream)
      .def("id", &DataChannel::id)
      .def("label", &DataChannel::label)
      .def("protocol", &DataChannel::protocol)
      .def("reliability", &DataChannel::reliability);
}

// ---- track.hpp ----

void bind_track(nb::module_& m) {
  nb::class_<Track, Channel>(m, "Track")
      .def("is_open", &Track::isOpen)
      .def("is_closed", &Track::isClosed)
      .def("max_message_size", &Track::maxMessageSize)
      .def("close", &Track::close)
      .def("send", nb::overload_cast<message_variant>(&Track::send), "data"_a)
      .def(
          "send",
          [](Track& self, std::vector<byte> data, size_t size) {
            return self.send(data.data(), size);
          },
          "data"_a, "size"_a)
      .def("mid", &Track::mid)
      .def("direction", &Track::direction)
      .def("description", &Track::description)
      .def("set_description", &Track::setDescription, "description"_a)
      .def("on_frame", &Track::onFrame, "callback"_a)
      .def("request_keyframe", &Track::requestKeyframe)
      .def("request_bitrate", &Track::requestBitrate, "bitrate"_a)
      .def("set_media_handler", &Track::setMediaHandler, "handler"_a)
      .def("chain_media_handler", &Track::chainMediaHandler, "handler"_a)
      .def("get_media_handler", &Track::getMediaHandler);
}

// ---- peerconnection.hpp ----

void bind_peerconnection(nb::module_& m) {
  nb::class_<DataChannelInit>(m, "DataChannelInit")
      .def(nb::init<>())
      .def_rw("reliability", &DataChannelInit::reliability)
      .def_rw("negotiated", &DataChannelInit::negotiated)
      .def_rw("id", &DataChannelInit::id)
      .def_rw("protocol", &DataChannelInit::protocol);

  nb::class_<LocalDescriptionInit>(m, "LocalDescriptionInit")
      .def(nb::init<>())
      .def_rw("ice_ufrag", &LocalDescriptionInit::iceUfrag)
      .def_rw("ice_pwd", &LocalDescriptionInit::icePwd);

  nb::class_<PeerConnection> pc(m, "PeerConnection");

  // PeerConnection 内の enum
  nb::enum_<PeerConnection::State>(pc, "State")
      .value("New", PeerConnection::State::New)
      .value("Connecting", PeerConnection::State::Connecting)
      .value("Connected", PeerConnection::State::Connected)
      .value("Disconnected", PeerConnection::State::Disconnected)
      .value("Failed", PeerConnection::State::Failed)
      .value("Closed", PeerConnection::State::Closed);

  nb::enum_<PeerConnection::IceState>(pc, "IceState")
      .value("New", PeerConnection::IceState::New)
      .value("Checking", PeerConnection::IceState::Checking)
      .value("Connected", PeerConnection::IceState::Connected)
      .value("Completed", PeerConnection::IceState::Completed)
      .value("Failed", PeerConnection::IceState::Failed)
      .value("Disconnected", PeerConnection::IceState::Disconnected)
      .value("Closed", PeerConnection::IceState::Closed);

  nb::enum_<PeerConnection::GatheringState>(pc, "GatheringState")
      .value("New", PeerConnection::GatheringState::New)
      .value("InProgress", PeerConnection::GatheringState::InProgress)
      .value("Complete", PeerConnection::GatheringState::Complete);

  nb::enum_<PeerConnection::SignalingState>(pc, "SignalingState")
      .value("Stable", PeerConnection::SignalingState::Stable)
      .value("HaveLocalOffer", PeerConnection::SignalingState::HaveLocalOffer)
      .value("HaveRemoteOffer", PeerConnection::SignalingState::HaveRemoteOffer)
      .value("HaveLocalPranswer",
             PeerConnection::SignalingState::HaveLocalPranswer)
      .value("HaveRemotePranswer",
             PeerConnection::SignalingState::HaveRemotePranswer);

  // PeerConnection
  pc.def(nb::init<>())
      .def(nb::init<Configuration>(), "config"_a)
      .def("close", &PeerConnection::close)
      .def("config", &PeerConnection::config, nb::rv_policy::reference)
      .def("state", &PeerConnection::state)
      .def("ice_state", &PeerConnection::iceState)
      .def("gathering_state", &PeerConnection::gatheringState)
      .def("signaling_state", &PeerConnection::signalingState)
      .def("negotiation_needed", &PeerConnection::negotiationNeeded)
      .def("has_media", &PeerConnection::hasMedia)
      .def("local_description", &PeerConnection::localDescription)
      .def("remote_description", &PeerConnection::remoteDescription)
      .def("remote_max_message_size", &PeerConnection::remoteMaxMessageSize)
      .def("local_address", &PeerConnection::localAddress)
      .def("remote_address", &PeerConnection::remoteAddress)
      .def("max_data_channel_id", &PeerConnection::maxDataChannelId)
      .def("get_selected_candidate_pair",
           &PeerConnection::getSelectedCandidatePair)
      .def("set_local_description", &PeerConnection::setLocalDescription,
           "type"_a = Description::Type::Unspec,
           "init"_a = LocalDescriptionInit{})
      .def("set_remote_description", &PeerConnection::setRemoteDescription)
      .def("add_remote_candidate", &PeerConnection::addRemoteCandidate)
      .def("gather_local_candidates", &PeerConnection::gatherLocalCandidates,
           "additional_ice_servers"_a = std::vector<IceServer>{})
      .def("set_media_handler", &PeerConnection::setMediaHandler)
      .def("get_media_handler", &PeerConnection::getMediaHandler)
      .def("create_data_channel", &PeerConnection::createDataChannel, "label"_a,
           "init"_a = DataChannelInit{})
      .def("on_data_channel", &PeerConnection::onDataChannel)
      .def("add_track", &PeerConnection::addTrack)
      .def("on_track", &PeerConnection::onTrack)
      .def("on_local_description", &PeerConnection::onLocalDescription)
      .def("on_local_candidate", &PeerConnection::onLocalCandidate)
      .def("on_state_change", &PeerConnection::onStateChange)
      .def("on_ice_state_change", &PeerConnection::onIceStateChange)
      .def("on_gathering_state_change", &PeerConnection::onGatheringStateChange)
      .def("on_signaling_state_change", &PeerConnection::onSignalingStateChange)
      .def("reset_callbacks", &PeerConnection::resetCallbacks)
      .def("remote_fingerprint", &PeerConnection::remoteFingerprint)
      .def("clear_stats", &PeerConnection::clearStats)
      .def("bytes_sent", &PeerConnection::bytesSent)
      .def("bytes_received", &PeerConnection::bytesReceived)
      .def("rtt", &PeerConnection::rtt);
}

// ---- websocket.hpp ----

void bind_websocket(nb::module_& m) {
  nb::class_<WebSocket, Channel> ws(m, "WebSocket");

  // WebSocket::State
  nb::enum_<WebSocket::State>(ws, "State")
      .value("Connecting", WebSocket::State::Connecting)
      .value("Open", WebSocket::State::Open)
      .value("Closing", WebSocket::State::Closing)
      .value("Closed", WebSocket::State::Closed);

  // WebSocket
  ws.def(nb::init<>())
      .def(nb::init<WebSocket::Configuration>(), "config"_a)
      .def("is_open", &WebSocket::isOpen)
      .def("is_closed", &WebSocket::isClosed)
      .def("max_message_size", &WebSocket::maxMessageSize)
      .def("close", &WebSocket::close)
      .def("send", nb::overload_cast<message_variant>(&WebSocket::send),
           "data"_a)
      .def(
          "send",
          [](WebSocket& self, std::vector<byte> data, size_t size) {
            return self.send(data.data(), size);
          },
          "data"_a, "size"_a)
      .def("ready_state", &WebSocket::readyState)
      .def("open", &WebSocket::open, "url"_a)
      .def("force_close", &WebSocket::forceClose)
      .def("remote_address", &WebSocket::remoteAddress)
      .def("path", &WebSocket::path);
}

// ---- websocketserver.hpp ----

void bind_websocketserver(nb::module_& m) {
  nb::class_<WebSocketServer>(m, "WebSocketServer")
      .def(nb::init<>())
      .def(nb::init<WebSocketServer::Configuration>(), "config"_a)
      .def("stop", &WebSocketServer::stop)
      .def("port", &WebSocketServer::port)
      .def("on_client", &WebSocketServer::onClient, "callback"_a);
}

}  // namespace

// ---- libdatachannel ----

void bind_libdatachannel(nb::module_& m) {
  bind_configuration(m);
  bind_description(m);
  bind_candidate(m);
  bind_reliability(m);
  bind_frameinfo(m);
  bind_message(m);
  bind_nalunit(m);
  bind_h265nalunit(m);
  bind_mediahandler(m);
  bind_rtppacketizationconfig(m);
  bind_rtppacketizer(m);
  bind_av1rtppacketizer(m);
  bind_h264rtppacketizer(m);
  bind_h265rtppacketizer(m);
  bind_rtpdepacketizer(m);
  bind_h264depacketizer(m);
  bind_pacinghandler(m);
  bind_rembhandler(m);
  bind_plihandler(m);
  bind_rtcpnackresponder(m);
  bind_rtcpreceivingsession(m);
  bind_rtcpsrreporter(m);
  bind_channel(m);
  bind_datachannel(m);
  bind_track(m);
  bind_peerconnection(m);
  bind_websocket(m);
  bind_websocketserver(m);
}
