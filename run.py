import argparse
import glob
import multiprocessing
import os
import shutil
import sys
from typing import Optional

from buildbase import (
    Platform,
    add_path,
    cd,
    cmake_path,
    cmd,
    cmdcap,
    get_macos_osver,
    get_windows_osver,
    git_clone_shallow,
    install_aom,
    install_cmake,
    install_libjpeg_turbo,
    install_libyuv,
    install_mbedtls,
    install_openh264,
    install_opus,
    mkdir_p,
    read_version_file,
)
from pypath import get_python_include_dir, get_python_library, get_python_version

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


def install_deps(
    platform: Platform,
    source_dir,
    build_dir,
    install_dir,
    debug,
):
    version = read_version_file("DEPS")

    # CMake
    install_cmake_args = {
        "version": version["CMAKE_VERSION"],
        "version_file": os.path.join(install_dir, "cmake.version"),
        "source_dir": source_dir,
        "install_dir": install_dir,
        "platform": "",
        "ext": "tar.gz",
    }
    if platform.build.os == "windows" and platform.build.arch == "x86_64":
        install_cmake_args["platform"] = "windows-x86_64"
        install_cmake_args["ext"] = "zip"
    elif platform.build.os == "macos":
        install_cmake_args["platform"] = "macos-universal"
    elif platform.build.os == "ubuntu" and platform.build.arch == "x86_64":
        install_cmake_args["platform"] = "linux-x86_64"
    elif platform.build.os == "ubuntu" and platform.build.arch == "armv8":
        install_cmake_args["platform"] = "linux-aarch64"
    else:
        raise Exception("Failed to install CMake")
    install_cmake(**install_cmake_args)

    if platform.build.os == "macos":
        add_path(os.path.join(install_dir, "cmake", "CMake.app", "Contents", "bin"))
    else:
        add_path(os.path.join(install_dir, "cmake", "bin"))

    # libdatachannel
    libdatachannel_dir = os.path.join(source_dir, "libdatachannel")
    libdatachannel_url = "https://github.com/paullouisageneau/libdatachannel.git"
    if not os.path.exists(os.path.join(libdatachannel_dir, ".git")):
        git_clone_shallow(
            libdatachannel_url,
            version["LIBDATACHANNEL_VERSION"],
            libdatachannel_dir,
            submodule=True,
        )

    macos_cmake_args = []
    if platform.target.os == "macos":
        sysroot = cmdcap(["xcrun", "--sdk", "macosx", "--show-sdk-path"])
        target = (
            "x86_64-apple-darwin" if platform.target.arch == "x86_64" else "aarch64-apple-darwin"
        )
        arch = "x86_64" if platform.target.arch == "x86_64" else "arm64"
        macos_cmake_args.append(f"-DCMAKE_SYSTEM_PROCESSOR={arch}")
        macos_cmake_args.append(f"-DCMAKE_OSX_ARCHITECTURES={arch}")
        macos_cmake_args.append(f"-DCMAKE_C_COMPILER_TARGET={target}")
        macos_cmake_args.append(f"-DCMAKE_CXX_COMPILER_TARGET={target}")
        macos_cmake_args.append(f"-DCMAKE_OBJCXX_COMPILER_TARGET={target}")
        macos_cmake_args.append(f"-DCMAKE_SYSROOT={sysroot}")

    # MbedTLS
    install_mbedtls_args = {
        "version": version["MBEDTLS_VERSION"],
        "version_file": os.path.join(install_dir, "mbedtls.version"),
        "source_dir": source_dir,
        "build_dir": build_dir,
        "install_dir": install_dir,
        "debug": debug,
        "cmake_args": macos_cmake_args,
    }
    install_mbedtls(**install_mbedtls_args)

    # OpenH264
    install_openh264_args = {
        "version": version["OPENH264_VERSION"],
        "version_file": os.path.join(install_dir, "openh264.version"),
        "source_dir": source_dir,
        "install_dir": install_dir,
        "is_windows": platform.target.os == "windows",
    }
    install_openh264(**install_openh264_args)

    # Opus
    install_opus_args = {
        "version": version["OPUS_VERSION"],
        "version_file": os.path.join(install_dir, "opus.version"),
        "source_dir": source_dir,
        "build_dir": build_dir,
        "install_dir": install_dir,
        "configuration": "Debug" if debug else "Release",
        "cmake_args": macos_cmake_args,
        "expected_sha256": version.get("OPUS_SHA256_HASH"),
    }
    install_opus(**install_opus_args)

    # libjpeg-turbo
    install_libjpeg_turbo_args = {
        "version": version["LIBJPEG_TURBO_VERSION"],
        "version_file": os.path.join(install_dir, "libjpeg-turbo.version"),
        "source_dir": source_dir,
        "build_dir": build_dir,
        "install_dir": install_dir,
        "configuration": "Debug" if debug else "Release",
        "cmake_args": macos_cmake_args,
    }
    install_libjpeg_turbo(**install_libjpeg_turbo_args)

    # libyuv
    install_libyuv_args = {
        "version": version["LIBYUV_VERSION"],
        "version_file": os.path.join(install_dir, "libyuv.version"),
        "source_dir": source_dir,
        "build_dir": build_dir,
        "install_dir": install_dir,
        "libjpeg_turbo_dir": os.path.join(install_dir, "libjpeg-turbo"),
        "configuration": "Debug" if debug else "Release",
        "cmake_args": macos_cmake_args,
    }
    install_libyuv(**install_libyuv_args)

    # AOM
    install_aom_args = {
        "version": version["AOM_VERSION"],
        "version_file": os.path.join(install_dir, "aom.version"),
        "source_dir": source_dir,
        "build_dir": build_dir,
        "install_dir": install_dir,
        "configuration": "Debug" if debug else "Release",
        "cmake_args": macos_cmake_args,
    }
    install_aom(**install_aom_args)


AVAILABLE_TARGETS = [
    "windows_x86_64",
    "macos_arm64",
    "ubuntu-22.04_x86_64",
    "ubuntu-24.04_x86_64",
    "ubuntu-24.04_armv8",
    "ubuntu-22.04_armv8_jetson",
]


def _find_clang_binary(name: str) -> Optional[str]:
    if shutil.which(name) is not None:
        return name
    else:
        for n in range(50, 14, -1):
            if shutil.which(f"{name}-{n}") is not None:
                return f"{name}-{n}"
    return None


def _format(
    clang_format_path: Optional[str] = None,
):
    if clang_format_path is None:
        clang_format_path = _find_clang_binary("clang-format")
    if clang_format_path is None:
        raise Exception("clang-format not found. Please install it or specify the path.")
    patterns = [
        "src/**/*.cpp",
        "src/**/*.h",
    ]
    target_files = []
    for pattern in patterns:
        files = glob.glob(pattern, recursive=True)
        target_files.extend(files)
    cmd([clang_format_path, "-i"] + target_files)


def _build(args):
    target = args.target
    if target == "windows_x86_64":
        platform = Platform("windows", get_windows_osver(), "x86_64")
    elif target == "macos_x86_64":
        platform = Platform("macos", get_macos_osver(), "x86_64")
    elif target == "macos_arm64":
        platform = Platform("macos", get_macos_osver(), "arm64")
    elif target == "ubuntu-22.04_x86_64":
        platform = Platform("ubuntu", "22.04", "x86_64")
    elif target == "ubuntu-24.04_x86_64":
        platform = Platform("ubuntu", "24.04", "x86_64")
    elif target == "ubuntu-24.04_armv8":
        platform = Platform("ubuntu", "24.04", "armv8")
    elif target == "ubuntu-22.04_armv8_jetson":
        platform = Platform("jetson", None, "armv8", "ubuntu-22.04")
    else:
        raise Exception(f"Unknown target {target}")

    source_dir = os.path.join(BASE_DIR, "_source", platform.target.package_name)
    build_dir = os.path.join(BASE_DIR, "_build", platform.target.package_name)
    install_dir = os.path.join(BASE_DIR, "_install", platform.target.package_name)
    mkdir_p(source_dir)
    mkdir_p(build_dir)
    mkdir_p(install_dir)

    with cd(BASE_DIR):
        install_deps(
            platform,
            source_dir,
            build_dir,
            install_dir,
            args.debug,
        )

        configuration = "Release"
        if args.debug:
            configuration = "Debug"
        if args.relwithdebinfo:
            configuration = "RelWithDebInfo"

        cmake_args = []
        cmake_args.append(f"-DCMAKE_BUILD_TYPE={configuration}")
        cmake_args.append(f"-DTARGET_OS={platform.target.os}")
        cmake_args.append("-DCMAKE_POLICY_VERSION_MINIMUM=3.5")
        cmake_args.append(
            f"-DLIBDATACHANNEL_DIR={cmake_path(os.path.join(source_dir, 'libdatachannel'))}"
        )
        python_version = get_python_version()
        cmake_args.append(f"-DPYTHON_VERSION_STRING={python_version}")
        cmake_args.append(f"-DPYTHON_INCLUDE_DIR={get_python_include_dir(python_version)}")
        cmake_args.append(f"-DPYTHON_EXECUTABLE={cmake_path(sys.executable)}")
        python_library = get_python_library(python_version)
        if python_library is None:
            raise Exception("Failed to get Python library")
        cmake_args.append(f"-DPYTHON_LIBRARY={cmake_path(python_library)}")

        # libdatachannel
        cmake_args.append("-DUSE_MBEDTLS=ON")
        cmake_args.append(f"-DMbedTLS_ROOT={cmake_path(os.path.join(install_dir, 'mbedtls'))}")
        cmake_args.append("-DUSE_GNUTLS=OFF")
        cmake_args.append("-DUSE_NICE=OFF")
        cmake_args.append("-DNO_TESTS=ON")
        cmake_args.append("-DNO_EXAMPLES=ON")

        # OpenH264
        cmake_args.append(f"-DOPENH264_DIR={cmake_path(os.path.join(install_dir, 'openh264'))}")

        # Opus
        cmake_args.append(f"-DOPUS_DIR={cmake_path(os.path.join(install_dir, 'opus'))}")

        # Libyuv
        cmake_args.append(
            f"-DLIBJPEG_TURBO_DIR={cmake_path(os.path.join(install_dir, 'libjpeg-turbo'))}"
        )
        cmake_args.append(f"-DLIBYUV_DIR={cmake_path(os.path.join(install_dir, 'libyuv'))}")

        # AOM
        cmake_args.append(f"-DAOM_DIR={cmake_path(os.path.join(install_dir, 'aom'))}")

        if platform.target.os == "macos":
            sysroot = cmdcap(["xcrun", "--sdk", "macosx", "--show-sdk-path"])
            cmake_args += [
                "-DCMAKE_SYSTEM_PROCESSOR=arm64",
                "-DCMAKE_OSX_ARCHITECTURES=arm64",
                "-DCMAKE_C_COMPILER=clang",
                "-DCMAKE_C_COMPILER_TARGET=aarch64-apple-darwin",
                "-DCMAKE_CXX_COMPILER=clang++",
                "-DCMAKE_CXX_COMPILER_TARGET=aarch64-apple-darwin",
                f"-DCMAKE_SYSROOT={sysroot}",
            ]

        # Windows 以外の、クロスコンパイルでない環境では pyi ファイルを生成する
        if (
            platform.target.os != "windows"
            and platform.build.package_name == platform.target.package_name
        ):
            cmake_args.append("-DLIBDATACHANNEL_PY_GEN_PYI=ON")

        libdatachannelpy_src_dir = os.path.join("src", "libdatachannel")
        libdatachannelpy_build_dir = os.path.join(build_dir, "libdatachannel-py")
        if platform.target.os == "windows":
            libdatachannelpy_build_target_dir = os.path.join(
                build_dir, "libdatachannel-py", configuration
            )
        else:
            libdatachannelpy_build_target_dir = os.path.join(build_dir, "libdatachannel-py")

        mkdir_p(libdatachannelpy_build_dir)
        with cd(libdatachannelpy_build_dir):
            cmd(["cmake", BASE_DIR, *cmake_args])
            cmd(
                [
                    "cmake",
                    "--build",
                    ".",
                    "--config",
                    configuration,
                    f"-j{multiprocessing.cpu_count()}",
                ]
            )

        for file in os.listdir(libdatachannelpy_src_dir):
            if file.startswith("libdatachannel_ext.") and (
                file.endswith(".so") or file.endswith(".dylib") or file.endswith(".pyd")
            ):
                os.remove(os.path.join(libdatachannelpy_src_dir, file))

        for file in os.listdir(libdatachannelpy_build_target_dir):
            if file.startswith("libdatachannel_ext.") and (
                file.endswith(".so") or file.endswith(".dylib") or file.endswith(".pyd")
            ):
                shutil.copyfile(
                    os.path.join(libdatachannelpy_build_target_dir, file),
                    os.path.join(libdatachannelpy_src_dir, file),
                )
            if file in ("libdatachannel_ext.pyi", "py.typed"):
                shutil.copyfile(
                    os.path.join(libdatachannelpy_build_target_dir, file),
                    os.path.join(libdatachannelpy_src_dir, file),
                )

        for file in os.listdir(os.path.join(libdatachannelpy_build_target_dir, "libyuv")):
            if file in ("__init__.pyi",):
                shutil.copyfile(
                    os.path.join(libdatachannelpy_build_target_dir, "libyuv", file),
                    os.path.join(libdatachannelpy_src_dir, "libyuv", file),
                )

        for file in os.listdir(os.path.join(libdatachannelpy_build_target_dir, "codec")):
            if file in ("__init__.pyi",):
                shutil.copyfile(
                    os.path.join(libdatachannelpy_build_target_dir, "codec", file),
                    os.path.join(libdatachannelpy_src_dir, "codec", file),
                )


def main():
    parser = argparse.ArgumentParser()
    sp = parser.add_subparsers(dest="command")

    # build コマンド
    bp = sp.add_parser("build")
    bp.add_argument("target", choices=AVAILABLE_TARGETS)
    bp.add_argument("--debug", action="store_true")
    bp.add_argument("--relwithdebinfo", action="store_true")

    # format コマンド
    fp = sp.add_parser("format")
    fp.add_argument("--clang-format-path", type=str, default=None)

    args = parser.parse_args()

    if args.command == "build":
        _build(args)
    elif args.command == "format":
        _format(clang_format_path=args.clang_format_path)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
