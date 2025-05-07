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
#include <nanobind/stl/chrono.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
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
      return false;
    }
    nb::bytes pybytes(src);
    size_t len = pybytes.size();
    const char* data = pybytes.c_str();
    value.assign(reinterpret_cast<const std::byte*>(data),
                 reinterpret_cast<const std::byte*>(data + len));
    return true;
  }

  static handle from_cpp(const std::vector<std::byte>& vec,
                         rv_policy policy,
                         cleanup_list* cleanup) {
    return nb::bytes(reinterpret_cast<const char*>(vec.data()), vec.size());
  }
};
}  // namespace detail
}  // namespace nanobind

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
           [](const Description& d) { return static_cast<std::string>(d); });

  // Entry（抽象基底）
  nb::class_<Description::Entry> entry(m, "Entry");

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
  nb::class_<Description::Application, Description::Entry>(m, "Application")
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
  nb::class_<Description::Media, Description::Entry>(m, "Media")
      .def(nb::init<const std::string&, std::string, Description::Direction>(),
           "mline"_a, "mid"_a, "direction"_a = Description::Direction::SendOnly)
      .def(nb::init<const std::string&>(), "sdp"_a)
      .def("description", &Description::Media::description)
      .def("reciprocate", &Description::Media::reciprocate)
      .def("add_ssrc", &Description::Media::addSSRC)
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
  nb::class_<Description::Media::RtpMap> rtpmap(m, "RtpMap");
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
  nb::class_<Description::Audio, Description::Media>(m, "Audio")
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
  nb::class_<Description::Video, Description::Media>(m, "Video")
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

  desc.def("has_application", &Description::hasApplication)
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
      .def("__eq__", &Candidate::operator==)
      .def("__ne__", &Candidate::operator!=)
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
          [](std::shared_ptr<MediaHandler> self, message_vector& messages,
             const message_callback& send) { self->incoming(messages, send); },
          "messages"_a, "send"_a)
      .def(
          "outgoing",
          [](std::shared_ptr<MediaHandler> self, message_vector& messages,
             const message_callback& send) { self->outgoing(messages, send); },
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

// ---- channel.hpp ----

void bind_channel(nb::module_& m) {
  nb::class_<Channel>(m, "Channel")
      // Core API
      .def("close", &Channel::close)
      .def("send", nb::overload_cast<message_variant>(&Channel::send), "data"_a)
      .def("send", nb::overload_cast<const byte*, size_t>(&Channel::send),
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
      .def("stream", &DataChannel::stream)
      .def("id", &DataChannel::id)
      .def("label", &DataChannel::label)
      .def("protocol", &DataChannel::protocol)
      .def("reliability", &DataChannel::reliability)
      .def("max_message_size", &DataChannel::maxMessageSize)
      .def("close", &DataChannel::close);
}

NB_MODULE(libdatachannel_ext, m) {
  bind_configuration(m);
  bind_description(m);
  bind_candidate(m);
  bind_reliability(m);
  bind_frameinfo(m);
  bind_message(m);
  bind_mediahandler(m);
  bind_rtppacketizationconfig(m);
  bind_rtppacketizer(m);
  bind_channel(m);
  bind_datachannel(m);

  nb::class_<PeerConnection>(m, "PeerConnection");
}
