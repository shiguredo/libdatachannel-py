#include <nanobind/nanobind.h>

extern void bind_libdatachannel(nanobind::module_& m);
extern void bind_codec(nanobind::module_& m);
extern void bind_libyuv(nanobind::module_& m);

NB_MODULE(libdatachannel_ext, m) {
  bind_libdatachannel(m);
  nanobind::module_ codec = m.def_submodule("codec");
  bind_codec(codec);
  nanobind::module_ libyuv = m.def_submodule("libyuv");
  bind_libyuv(libyuv);
}
