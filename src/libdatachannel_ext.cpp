#include <nanobind/nanobind.h>

extern void bind_libdatachannel(nanobind::module_& m);

NB_MODULE(libdatachannel_ext, m) {
  bind_libdatachannel(m);
}
