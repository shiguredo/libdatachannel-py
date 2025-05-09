cmake_minimum_required(VERSION 3.23)

project(libdatachannel-py)

# Only interpret if() arguments as variables or keywords when unquoted.
cmake_policy(SET CMP0054 NEW)
# MSVC runtime library flags are selected by an abstraction.
cmake_policy(SET CMP0091 NEW)

if (NOT CMAKE_CROSSCOMPILING)
  set(Python_VERSION "${PYTHON_VERSION_STRING}")
  set(Python_EXECUTABLE "${PYTHON_EXECUTABLE}")
  set(Python_INCLUDE_DIR "${PYTHON_INCLUDE_DIR}")
  set(Python_LIBRARIES "${PYTHON_LIBRARY}")
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
endif()

if (CMAKE_CROSSCOMPILING)
  set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY NEVER)
  set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE NEVER)
  set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE NEVER)
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
  set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)
  set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE BOTH)
  set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE BOTH)
endif()

message(STATUS "-- Python_FOUND: ${Python_FOUND}")
message(STATUS "-- Python_Development.Module_FOUND: ${Python_Development.Module_FOUND}")
message(STATUS "-- Python_EXECUTABLE: ${Python_EXECUTABLE}")
message(STATUS "-- Python_INCLUDE_DIR: ${Python_INCLUDE_DIR}")
message(STATUS "-- Python_INCLUDE_DIRS: ${Python_INCLUDE_DIRS}")
message(STATUS "-- Python_LIBRARIES: ${Python_LIBRARIES}")
message(STATUS "-- Python_LIBRARY_DIRS: ${Python_LIBRARY_DIRS}")
message(STATUS "-- Python_RUNTIME_LIBRARY_DIRS: ${Python_RUNTIME_LIBRARY_DIRS}")

if (NB_CMAKE_DIR)
  list(APPEND CMAKE_PREFIX_PATH "${NB_CMAKE_DIR}")
else()
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)
  list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
endif()

set(TARGET_OS "" CACHE STRING "ビルド対象の動作する OS。\n有効な値は windows, macos, ubuntu")
set(LIBDATACHANNEL_DIR "" CACHE PATH "libdatachannel のルートディレクトリ")

find_package(nanobind CONFIG REQUIRED)

add_subdirectory(${LIBDATACHANNEL_DIR} libdatachannel EXCLUDE_FROM_ALL)
add_subdirectory(${LIBDATACHANNEL_DIR}/deps/json json EXCLUDE_FROM_ALL)

nanobind_add_module(
  libdatachannel_ext
  NB_STATIC
  src/libdatachannel_ext.cpp
)

if (LIBDATACHANNEL_PY_GEN_PYI)
  nanobind_add_stub(
    libdatachannel_ext_stub
    MODULE libdatachannel_ext
    OUTPUT libdatachannel_ext.pyi
    PYTHON_PATH $<TARGET_FILE_DIR:libdatachannel_ext>
    DEPENDS libdatachannel_ext
    MARKER_FILE py.typed
  )
endif()

set_target_properties(libdatachannel_ext PROPERTIES CXX_STANDARD 20 C_STANDARD 20)
set_target_properties(libdatachannel_ext PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(TARGET_OS STREQUAL "macos")
  set_target_properties(libdatachannel_ext PROPERTIES CXX_VISIBILITY_PRESET hidden)
elseif(TARGET_OS STREQUAL "ubuntu")
elseif(TARGET_OS STREQUAL "jetson")
elseif(TARGET_OS STREQUAL "windows")
  # 文字コードを utf-8 として扱うのと、シンボルテーブル数を増やす
  target_compile_options(libdatachannel_ext PRIVATE /utf-8 /bigobj)
  # CRTライブラリを静的リンクさせる
  set_property(TARGET libdatachannel_ext PROPERTY
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  set_property(TARGET nanobind-static PROPERTY
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  target_compile_definitions(libdatachannel_ext
    PRIVATE
      _CONSOLE
      _WIN32_WINNT=0x0A00
      NOMINMAX
      WIN32_LEAN_AND_MEAN
      HAVE_SNPRINTF
  )
endif()

target_link_libraries(libdatachannel_ext PRIVATE LibDataChannel::LibDataChannelStatic)
target_link_libraries(nanobind-static PRIVATE LibDataChannel::LibDataChannelStatic)

install(TARGETS libdatachannel_ext LIBRARY DESTINATION .)
if (LIBDATACHANNEL_PY_GEN_PYI)
  install(FILES py.typed libdatachannel_ext.pyi DESTINATION ".")
endif()
