cmake_minimum_required(VERSION 3.23)

# Only interpret if() arguments as variables or keywords when unquoted.
cmake_policy(SET CMP0054 NEW)
# MSVC runtime library flags are selected by an abstraction.
cmake_policy(SET CMP0091 NEW)

# CMake 4 以降では 3.5 未満の互換モードが削除されているため、
# 古いサブプロジェクトの cmake_minimum_required(VERSION 2.x) などで
# 失敗しないよう最低ポリシーバージョンを 3.5 に固定する
if(NOT DEFINED CMAKE_POLICY_VERSION_MINIMUM OR "${CMAKE_POLICY_VERSION_MINIMUM}" STREQUAL "")
  set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
endif()

project(libdatachannel-py)

# run.py が生成するヒントファイルを読み込む（存在すれば）
if(EXISTS "${CMAKE_SOURCE_DIR}/build_hints.cmake")
  include("${CMAKE_SOURCE_DIR}/build_hints.cmake")
endif()

#[[
  run.py で依存を事前構築した後に uv build だけで完結させるため、
  未指定の依存パスを _install/_source 配下から自動推測する。
  ただし editable/sdist 段階では走らせず、wheel 段階のみ有効にする。
]]
if("$ENV{SKBUILD_STATE}" STREQUAL "wheel")
  set(_AUTO_DIR "")
  if(APPLE)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
      set(_AUTO_DIR "macos_arm64")
    else()
      set(_AUTO_DIR "macos_x86_64")
    endif()
  elseif(WIN32)
    set(_AUTO_DIR "windows_x86_64")
  elseif(UNIX)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64")
      foreach(d "ubuntu-24.04_x86_64" "ubuntu-22.04_x86_64")
        if(EXISTS "${CMAKE_SOURCE_DIR}/_install/${d}")
          set(_AUTO_DIR "${d}")
          break()
        endif()
      endforeach()
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64|armv8")
      foreach(d "ubuntu-24.04_armv8" "ubuntu-22.04_armv8_jetson")
        if(EXISTS "${CMAKE_SOURCE_DIR}/_install/${d}")
          set(_AUTO_DIR "${d}")
          break()
        endif()
      endforeach()
    endif()
  endif()

  if(NOT "${_AUTO_DIR}" STREQUAL "")
    set(_AUTO_INSTALL_DIR "${CMAKE_SOURCE_DIR}/_install/${_AUTO_DIR}")
    set(_AUTO_SOURCE_DIR  "${CMAKE_SOURCE_DIR}/_source/${_AUTO_DIR}")

    macro(define_if_empty var value)
      if(NOT DEFINED ${var} OR "${${var}}" STREQUAL "")
        set(${var} "${value}" CACHE PATH "" FORCE)
      endif()
    endmacro()

    define_if_empty(LIBDATACHANNEL_DIR "${_AUTO_SOURCE_DIR}/libdatachannel")
    define_if_empty(MbedTLS_ROOT         "${_AUTO_INSTALL_DIR}/mbedtls")
    define_if_empty(OPENH264_DIR         "${_AUTO_INSTALL_DIR}/openh264")
    define_if_empty(OPUS_DIR             "${_AUTO_INSTALL_DIR}/opus")
    define_if_empty(LIBJPEG_TURBO_DIR    "${_AUTO_INSTALL_DIR}/libjpeg-turbo")
    define_if_empty(LIBYUV_DIR           "${_AUTO_INSTALL_DIR}/libyuv")
    define_if_empty(AOM_DIR              "${_AUTO_INSTALL_DIR}/aom")

    if(NOT DEFINED TARGET_OS OR "${TARGET_OS}" STREQUAL "")
      if(APPLE)
        set(TARGET_OS "macos" CACHE STRING "" FORCE)
      elseif(WIN32)
        set(TARGET_OS "windows" CACHE STRING "" FORCE)
      else()
        if(_AUTO_DIR MATCHES "jetson")
          set(TARGET_OS "jetson" CACHE STRING "" FORCE)
        else()
          set(TARGET_OS "ubuntu" CACHE STRING "" FORCE)
        endif()
      endif()
    endif()
  endif()
endif()

#[[ 
  scikit-build-core が編集可能インストールやメタデータ生成の段階で
  CMake の configure を実行するため、依存（Opus など）が未指定の場合に
  失敗しないよう、拡張ビルドをスキップできるガードを入れる。
]]
# 依存が指定されているかをざっくり判定
set(_ALL_DEPS_PROVIDED TRUE)
foreach(_var IN ITEMS LIBDATACHANNEL_DIR OPUS_DIR LIBJPEG_TURBO_DIR LIBYUV_DIR AOM_DIR OPENH264_DIR)
  if(NOT DEFINED ${_var} OR "${${_var}}" STREQUAL "")
    set(_ALL_DEPS_PROVIDED FALSE)
  endif()
endforeach()

option(LIBDATACHANNEL_PY_REQUIRE_DEPS "依存が無い場合はエラーにする" OFF)

if(NOT _ALL_DEPS_PROVIDED)
  if(LIBDATACHANNEL_PY_REQUIRE_DEPS)
    message(FATAL_ERROR "Required dependencies are missing. Set *_DIR variables or provide CMAKE_ARGS.")
  else()
    if(DEFINED LIBDATACHANNEL_PY_PREBUILT_EXT AND EXISTS "${LIBDATACHANNEL_PY_PREBUILT_EXT}")
      message(STATUS "Using prebuilt extension: ${LIBDATACHANNEL_PY_PREBUILT_EXT}")
      # 拡張本体
      install(FILES "${LIBDATACHANNEL_PY_PREBUILT_EXT}" DESTINATION libdatachannel)
      # stub / pyi
      if(DEFINED LIBDATACHANNEL_PY_PREBUILT_PYI AND EXISTS "${LIBDATACHANNEL_PY_PREBUILT_PYI}")
        install(FILES "${LIBDATACHANNEL_PY_PREBUILT_PYI}" DESTINATION libdatachannel)
      endif()
      if(DEFINED LIBDATACHANNEL_PY_PREBUILT_CODEC_PYI AND EXISTS "${LIBDATACHANNEL_PY_PREBUILT_CODEC_PYI}")
        install(FILES "${LIBDATACHANNEL_PY_PREBUILT_CODEC_PYI}" DESTINATION libdatachannel/codec RENAME "__init__.pyi")
      endif()
      if(DEFINED LIBDATACHANNEL_PY_PREBUILT_LIBYUV_PYI AND EXISTS "${LIBDATACHANNEL_PY_PREBUILT_LIBYUV_PYI}")
        install(FILES "${LIBDATACHANNEL_PY_PREBUILT_LIBYUV_PYI}" DESTINATION libdatachannel/libyuv RENAME "__init__.pyi")
      endif()
      # パッケージの .py 類
      install(
        DIRECTORY src/libdatachannel/
        DESTINATION libdatachannel
        FILES_MATCHING
          PATTERN "*.py"
          PATTERN "py.typed"
      )
      return()
    else()
      message(STATUS "Dependencies not provided; skipping extension build (metadata/editable phase)")
      # Python パッケージの .py / .pyi / py.typed を同梱（wheel 側での配置用）
      install(
        DIRECTORY src/libdatachannel/
        DESTINATION libdatachannel
        FILES_MATCHING
          PATTERN "*.py"
          PATTERN "*.pyi"
          PATTERN "py.typed"
      )
      return()
    endif()
  endif()
endif()

list(APPEND CMAKE_PREFIX_PATH
  ${OPUS_DIR}
  ${LIBJPEG_TURBO_DIR}
)

if (NOT CMAKE_CROSSCOMPILING)
  if(DEFINED PYTHON_VERSION_STRING AND NOT "${PYTHON_VERSION_STRING}" STREQUAL "")
    set(Python_VERSION "${PYTHON_VERSION_STRING}")
  endif()
  if(DEFINED PYTHON_EXECUTABLE AND NOT "${PYTHON_EXECUTABLE}" STREQUAL "")
    set(Python_EXECUTABLE "${PYTHON_EXECUTABLE}")
  endif()
  if(DEFINED PYTHON_INCLUDE_DIR AND NOT "${PYTHON_INCLUDE_DIR}" STREQUAL "")
    set(Python_INCLUDE_DIR "${PYTHON_INCLUDE_DIR}")
  endif()
  if(DEFINED PYTHON_LIBRARY AND NOT "${PYTHON_LIBRARY}" STREQUAL "")
    set(Python_LIBRARIES "${PYTHON_LIBRARY}")
  endif()
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
find_package(Opus REQUIRED)
find_package(libjpeg-turbo REQUIRED)

add_subdirectory(${LIBDATACHANNEL_DIR} libdatachannel EXCLUDE_FROM_ALL)

nanobind_add_module(
  libdatachannel_ext
  NB_STATIC
  src/aom_video_decoder.cpp
  src/aom_video_encoder.cpp
  src/audio_codec.cpp
  src/bind_codec.cpp
  src/bind_libdatachannel.cpp
  src/bind_libyuv.cpp
  src/libdatachannel_ext.cpp
  src/openh264_video_encoder.cpp
  src/openh264_video_decoder.cpp
  src/opus_audio_encoder.cpp
  src/opus_audio_decoder.cpp
  src/video_codec.cpp
  src/videotoolbox_video_encoder.cpp
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
  nanobind_add_stub(
    libdatachannel_ext_codec_stub
    MODULE libdatachannel_ext.codec
    OUTPUT codec/__init__.pyi
    PYTHON_PATH $<TARGET_FILE_DIR:libdatachannel_ext>
    DEPENDS libdatachannel_ext
  )
  nanobind_add_stub(
    libdatachannel_ext_libyuv_stub
    MODULE libdatachannel_ext.libyuv
    OUTPUT libyuv/__init__.pyi
    PYTHON_PATH $<TARGET_FILE_DIR:libdatachannel_ext>
    DEPENDS libdatachannel_ext
  )
endif()

set_target_properties(libdatachannel_ext PROPERTIES CXX_STANDARD 20 C_STANDARD 20)
set_target_properties(libdatachannel_ext PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(TARGET_OS STREQUAL "macos")
  set_target_properties(libdatachannel_ext PROPERTIES CXX_VISIBILITY_PRESET hidden)
  target_link_libraries(libdatachannel_ext
    PRIVATE
      "-framework CoreFoundation"
      "-framework CoreVideo"
      "-framework CoreMedia"
      "-framework VideoToolbox"
  )
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

target_include_directories(libdatachannel_ext
  PRIVATE
    ${OPENH264_DIR}/include
)
target_link_libraries(libdatachannel_ext
  PRIVATE
    LibDataChannel::LibDataChannelStatic
    Opus::opus
    $<BUILD_INTERFACE:plog::plog>
)
target_link_libraries(nanobind-static PRIVATE LibDataChannel::LibDataChannelStatic)

# libyuv には cmake 用の設定が無いので直接指定する
target_include_directories(libdatachannel_ext
  PRIVATE
    ${LIBYUV_DIR}/include
)
target_link_libraries(libdatachannel_ext
  PRIVATE
    ${LIBYUV_DIR}/lib/libyuv.a
    libjpeg-turbo::turbojpeg-static
)

# AOM も同様
target_include_directories(libdatachannel_ext
  PRIVATE
    ${AOM_DIR}/include
)
target_link_libraries(libdatachannel_ext
  PRIVATE
    ${AOM_DIR}/lib/libaom.a
)

install(TARGETS libdatachannel_ext
  LIBRARY DESTINATION libdatachannel
  RUNTIME DESTINATION libdatachannel
)
if (LIBDATACHANNEL_PY_GEN_PYI)
  install(FILES py.typed libdatachannel_ext.pyi DESTINATION "libdatachannel")
endif()

# Python パッケージの .py / .pyi / py.typed を同梱
install(
  DIRECTORY src/libdatachannel/
  DESTINATION libdatachannel
  FILES_MATCHING
    PATTERN "*.py"
    PATTERN "*.pyi"
    PATTERN "py.typed"
)
