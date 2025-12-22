cmake_minimum_required(VERSION 4.2)
project(libdatachannel_py LANGUAGES CXX)

# プラットフォームチェック
if(NOT APPLE AND NOT UNIX AND NOT WIN32)
  message(FATAL_ERROR "This package supports macOS, Linux (Ubuntu), and Windows.")
endif()

# ExternalProject の依存関係は _deps 配下で管理
if(DEFINED SKBUILD_PROJECT_DIR)
  set(PROJECT_ROOT "${SKBUILD_PROJECT_DIR}")
else()
  set(PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
endif()

set(DEPS_DIR "${PROJECT_ROOT}/_deps")
message(STATUS "Dependencies directory: ${DEPS_DIR}")

# Python / nanobind
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

# 依存関係用の ExternalProject
include(ExternalProject)

# JSON から依存関係のバージョンを読み込み
file(READ ${CMAKE_CURRENT_SOURCE_DIR}/deps.json DEPS_JSON)

# MbedTLS の設定を解析
string(JSON MBEDTLS_VERSION GET ${DEPS_JSON} mbedtls version)
string(JSON MBEDTLS_GIT_REPOSITORY GET ${DEPS_JSON} mbedtls repository)
set(MBEDTLS_GIT_TAG "v${MBEDTLS_VERSION}")

# libdatachannel の設定を解析
string(JSON LIBDATACHANNEL_VERSION GET ${DEPS_JSON} libdatachannel version)
string(JSON LIBDATACHANNEL_GIT_REPOSITORY GET ${DEPS_JSON} libdatachannel repository)
set(LIBDATACHANNEL_GIT_TAG "v${LIBDATACHANNEL_VERSION}")

# ExternalProject ビルド用の並列ビルドジョブ数を決定
if(DEFINED ENV{CMAKE_BUILD_PARALLEL_LEVEL})
  set(PARALLEL_JOBS "$ENV{CMAKE_BUILD_PARALLEL_LEVEL}")
else()
  include(ProcessorCount)
  ProcessorCount(NCORES)
  if(NOT NCORES OR NCORES EQUAL 0)
    set(NCORES 2)
  endif()
  set(PARALLEL_JOBS "${NCORES}")
endif()

# ========== MbedTLS ==========
message(STATUS "Setting up MbedTLS...")
set(MBEDTLS_SOURCE_DIR "${DEPS_DIR}/mbedtls/${MBEDTLS_VERSION}/source")
set(MBEDTLS_BUILD_DIR "${DEPS_DIR}/mbedtls/${MBEDTLS_VERSION}/build")
set(MBEDTLS_INSTALL_DIR "${DEPS_DIR}/mbedtls/${MBEDTLS_VERSION}/install")

# Windows では .lib、それ以外では .a
if(WIN32)
  set(MBEDTLS_LIB ${MBEDTLS_INSTALL_DIR}/lib/mbedtls.lib)
else()
  set(MBEDTLS_LIB ${MBEDTLS_INSTALL_DIR}/lib/libmbedtls.a)
endif()

# ビルド済みかチェック
if(EXISTS ${MBEDTLS_LIB})
  message(STATUS "MbedTLS already built: ${MBEDTLS_LIB}")
  add_custom_target(mbedtls_build)
else()
  message(STATUS "Building MbedTLS...")
  ExternalProject_Add(
    mbedtls_build
    GIT_REPOSITORY ${MBEDTLS_GIT_REPOSITORY}
    GIT_TAG ${MBEDTLS_GIT_TAG}
    GIT_SHALLOW TRUE
    SOURCE_DIR ${MBEDTLS_SOURCE_DIR}
    BINARY_DIR ${MBEDTLS_BUILD_DIR}
    STAMP_DIR ${MBEDTLS_BUILD_DIR}/stamp
    TMP_DIR ${MBEDTLS_BUILD_DIR}/tmp
    UPDATE_COMMAND ""
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    CONFIGURE_COMMAND ${CMAKE_COMMAND} -E chdir ${MBEDTLS_SOURCE_DIR} ${Python_EXECUTABLE} scripts/config.py set MBEDTLS_SSL_DTLS_SRTP
      COMMAND ${CMAKE_COMMAND}
        -S ${MBEDTLS_SOURCE_DIR}
        -B ${MBEDTLS_BUILD_DIR}
        -DCMAKE_INSTALL_PREFIX=${MBEDTLS_INSTALL_DIR}
        -DCMAKE_BUILD_TYPE=Release
        -DENABLE_TESTING=OFF
        -DENABLE_PROGRAMS=OFF
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    BUILD_COMMAND ${CMAKE_COMMAND} --build ${MBEDTLS_BUILD_DIR} --config Release --parallel ${PARALLEL_JOBS}
    INSTALL_COMMAND ${CMAKE_COMMAND} --build ${MBEDTLS_BUILD_DIR} --config Release --target install
    BUILD_BYPRODUCTS
      ${MBEDTLS_LIB}
      ${MBEDTLS_INSTALL_DIR}/lib/libmbedx509.a
      ${MBEDTLS_INSTALL_DIR}/lib/libmbedcrypto.a
  )
endif()

# ========== libdatachannel ==========
message(STATUS "Setting up libdatachannel...")
set(LIBDATACHANNEL_SOURCE_DIR "${DEPS_DIR}/libdatachannel/${LIBDATACHANNEL_VERSION}/source")
set(LIBDATACHANNEL_BUILD_DIR "${DEPS_DIR}/libdatachannel/${LIBDATACHANNEL_VERSION}/build")
set(LIBDATACHANNEL_INSTALL_DIR "${DEPS_DIR}/libdatachannel/${LIBDATACHANNEL_VERSION}/install")

# Windows では .lib、それ以外では .a
if(WIN32)
  set(LIBDATACHANNEL_LIB ${LIBDATACHANNEL_INSTALL_DIR}/lib/datachannel.lib)
else()
  set(LIBDATACHANNEL_LIB ${LIBDATACHANNEL_INSTALL_DIR}/lib/libdatachannel.a)
endif()

# ビルド済みかチェック
if(EXISTS ${LIBDATACHANNEL_LIB})
  message(STATUS "libdatachannel already built: ${LIBDATACHANNEL_LIB}")
  add_custom_target(libdatachannel_build)
else()
  message(STATUS "Building libdatachannel...")
  ExternalProject_Add(
    libdatachannel_build
    GIT_REPOSITORY ${LIBDATACHANNEL_GIT_REPOSITORY}
    GIT_TAG ${LIBDATACHANNEL_GIT_TAG}
    GIT_SHALLOW TRUE
    GIT_SUBMODULES_RECURSE TRUE
    SOURCE_DIR ${LIBDATACHANNEL_SOURCE_DIR}
    BINARY_DIR ${LIBDATACHANNEL_BUILD_DIR}
    STAMP_DIR ${LIBDATACHANNEL_BUILD_DIR}/stamp
    TMP_DIR ${LIBDATACHANNEL_BUILD_DIR}/tmp
    UPDATE_COMMAND ""
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    CMAKE_ARGS
      -DCMAKE_INSTALL_PREFIX=${LIBDATACHANNEL_INSTALL_DIR}
      -DCMAKE_BUILD_TYPE=Release
      -DBUILD_SHARED_LIBS=OFF
      -DUSE_MBEDTLS=ON
      -DMbedTLS_ROOT=${MBEDTLS_INSTALL_DIR}
      -DUSE_GNUTLS=OFF
      -DUSE_NICE=OFF
      -DNO_TESTS=ON
      -DNO_EXAMPLES=ON
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON
      -DCMAKE_POLICY_VERSION_MINIMUM=3.5
    BUILD_COMMAND ${CMAKE_COMMAND} --build ${LIBDATACHANNEL_BUILD_DIR} --config Release --parallel ${PARALLEL_JOBS}
    INSTALL_COMMAND ${CMAKE_COMMAND} --build ${LIBDATACHANNEL_BUILD_DIR} --config Release --target install
    BUILD_BYPRODUCTS
      ${LIBDATACHANNEL_LIB}
      ${LIBDATACHANNEL_INSTALL_DIR}/lib/libjuice.a
      ${LIBDATACHANNEL_INSTALL_DIR}/lib/libsrtp2.a
      ${LIBDATACHANNEL_INSTALL_DIR}/lib/libusrsctp.a
    DEPENDS mbedtls_build
  )
endif()

# nanobind モジュール
nanobind_add_module(libdatachannel_ext
  NB_DOMAIN "libdatachannel"
  src/bind_libdatachannel.cpp
  src/libdatachannel_ext.cpp
)

# ビルド順序を保証するため依存関係を追加
add_dependencies(libdatachannel_ext mbedtls_build libdatachannel_build)

# インクルードディレクトリ
target_include_directories(libdatachannel_ext PRIVATE
  ${LIBDATACHANNEL_INSTALL_DIR}/include
  ${LIBDATACHANNEL_SOURCE_DIR}/deps/plog/include
)

# ライブラリをリンク
# libdatachannel とその依存ライブラリ、MbedTLS とその依存ライブラリをリンク
if(WIN32)
  target_link_libraries(libdatachannel_ext PRIVATE
    "${LIBDATACHANNEL_LIB}"
    "${LIBDATACHANNEL_INSTALL_DIR}/lib/libjuice.lib"
    "${LIBDATACHANNEL_INSTALL_DIR}/lib/srtp2.lib"
    "${LIBDATACHANNEL_INSTALL_DIR}/lib/usrsctp.lib"
    "${MBEDTLS_LIB}"
    "${MBEDTLS_INSTALL_DIR}/lib/mbedx509.lib"
    "${MBEDTLS_INSTALL_DIR}/lib/mbedcrypto.lib"
  )
else()
  target_link_libraries(libdatachannel_ext PRIVATE
    ${LIBDATACHANNEL_LIB}
    ${LIBDATACHANNEL_INSTALL_DIR}/lib/libjuice.a
    ${LIBDATACHANNEL_INSTALL_DIR}/lib/libsrtp2.a
    ${LIBDATACHANNEL_INSTALL_DIR}/lib/libusrsctp.a
    ${MBEDTLS_LIB}
    ${MBEDTLS_INSTALL_DIR}/lib/libmbedx509.a
    ${MBEDTLS_INSTALL_DIR}/lib/libmbedcrypto.a
  )
endif()

# プラットフォーム固有の設定
if(APPLE)
  target_link_libraries(libdatachannel_ext
    PRIVATE
      "-framework CoreFoundation"
  )
elseif(WIN32)
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

# コンパイルオプション
if(NOT WIN32)
  target_compile_options(libdatachannel_ext PRIVATE -fvisibility=hidden)
endif()

set_target_properties(libdatachannel_ext PROPERTIES OUTPUT_NAME "libdatachannel_ext")

# 型スタブ (.pyi) ファイルの自動生成
nanobind_add_stub(
  libdatachannel_stub
  MODULE libdatachannel_ext
  OUTPUT __init__.pyi
  PYTHON_PATH $<TARGET_FILE_DIR:libdatachannel_ext>
  DEPENDS libdatachannel_ext
  MARKER_FILE py.typed
)

# Wheel に配置するためのインストール先を指定
set(PY_PACKAGE "libdatachannel")
install(TARGETS libdatachannel_ext
  LIBRARY DESTINATION ${PY_PACKAGE}
  RUNTIME DESTINATION ${PY_PACKAGE}
)

# 型スタブファイルとマーカーファイルをインストール
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/__init__.pyi DESTINATION ${PY_PACKAGE})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/py.typed DESTINATION ${PY_PACKAGE})
