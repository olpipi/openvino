# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Build liburing (https://github.com/axboe/liburing) as a static library from
# the in-tree git submodule. Linux-only; on other platforms the file is
# not included from the top-level thirdparty/dependencies.cmake.

if(NOT LINUX)
    return()
endif()

set(LIBURING_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/liburing")
if(NOT EXISTS "${LIBURING_SOURCE_DIR}/src/liburing.map")
    message(FATAL_ERROR
        "liburing submodule is missing at ${LIBURING_SOURCE_DIR}. "
        "Run: git submodule update --init --recursive thirdparty/liburing/liburing")
endif()

set(LIBURING_SRC_DIR "${LIBURING_SOURCE_DIR}/src")
set(LIBURING_INCLUDE_DIR "${LIBURING_SRC_DIR}/include")
set(LIBURING_GEN_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/include")

# liburing's autotools-style configure script generates two private headers.
# For the OpenVINO prototype we synthesize minimal, kernel-headers-driven
# versions of them so the library can be built directly via CMake without
# invoking ./configure.
file(MAKE_DIRECTORY "${LIBURING_GEN_INCLUDE_DIR}/liburing")

set(LIBURING_COMPAT_H "${LIBURING_GEN_INCLUDE_DIR}/liburing/compat.h")
file(WRITE "${LIBURING_COMPAT_H}"
"/* Auto-generated minimal compat.h for OpenVINO build of liburing */
#ifndef LIBURING_COMPAT_H
#define LIBURING_COMPAT_H

#include <inttypes.h>
#include <linux/time_types.h>
#include <linux/openat2.h>

/* Block-layer io_uring_cmd opcodes used by io_uring_prep_cmd_discard().
 * These constants live in <linux/blkdev.h> only on very recent kernel headers
 * (>= 6.10). Provide upstream-matching fallbacks so the library compiles
 * against older kernel headers; the actual ioctl is dispatched by the kernel
 * at runtime, so the numeric values only need to match the UAPI definition. */
#ifndef BLOCK_URING_CMD_DISCARD
#define BLOCK_URING_CMD_DISCARD 0
#endif

#endif /* LIBURING_COMPAT_H */
")

# Parse the version from the upstream liburing.spec (or fall back to the tag).
set(LIBURING_VERSION_MAJOR 2)
set(LIBURING_VERSION_MINOR 14)
if(EXISTS "${LIBURING_SOURCE_DIR}/liburing.spec")
    file(STRINGS "${LIBURING_SOURCE_DIR}/liburing.spec" _spec_version
         REGEX "^Version:")
    if(_spec_version MATCHES "([0-9]+)\\.([0-9]+)")
        set(LIBURING_VERSION_MAJOR ${CMAKE_MATCH_1})
        set(LIBURING_VERSION_MINOR ${CMAKE_MATCH_2})
    endif()
endif()

set(LIBURING_VERSION_H "${LIBURING_GEN_INCLUDE_DIR}/liburing/io_uring_version.h")
file(WRITE "${LIBURING_VERSION_H}"
"/* Auto-generated io_uring_version.h for OpenVINO build of liburing */
#ifndef LIBURING_VERSION_H
#define LIBURING_VERSION_H

#define IO_URING_VERSION_MAJOR ${LIBURING_VERSION_MAJOR}
#define IO_URING_VERSION_MINOR ${LIBURING_VERSION_MINOR}

#endif /* LIBURING_VERSION_H */
")

# Collect all C sources from src/ and skip files that are optional or require
# autotools-detected features we deliberately do not enable in this build.
file(GLOB _liburing_srcs CONFIGURE_DEPENDS "${LIBURING_SRC_DIR}/*.c")
# sanitize.c pulls in ucontext / dlsym based instrumentation that the upstream
# configure script gates behind --enable-sanitizer; safe to drop here.
list(FILTER _liburing_srcs EXCLUDE REGEX "/sanitize\\.c$")
# nolibc helpers - not needed when linking against glibc/musl.
list(FILTER _liburing_srcs EXCLUDE REGEX "/nolibc\\.c$")

set(TARGET_NAME openvino_liburing)
add_library(${TARGET_NAME} STATIC ${_liburing_srcs})
add_library(openvino::liburing ALIAS ${TARGET_NAME})

target_include_directories(${TARGET_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${LIBURING_INCLUDE_DIR}>
        $<BUILD_INTERFACE:${LIBURING_GEN_INCLUDE_DIR}>
    PRIVATE
        ${LIBURING_SRC_DIR}
)

# liburing relies on a couple of GNU/Linux specific compile flags. Match the
# minimal set used by upstream's Makefile (CPPFLAGS/CFLAGS) without forcing the
# full warning-as-error policy that thirdparty must not inherit from OpenVINO.
target_compile_definitions(${TARGET_NAME} PRIVATE
    _GNU_SOURCE
    _LARGEFILE_SOURCE
    _FILE_OFFSET_BITS=64
    LIBURING_INTERNAL
)
target_compile_options(${TARGET_NAME} PRIVATE
    -Wno-unused-parameter
    -Wno-sign-compare
    -Wno-shift-negative-value
    -fno-strict-aliasing
)
set_target_properties(${TARGET_NAME} PROPERTIES
    C_VISIBILITY_PRESET hidden
    POSITION_INDEPENDENT_CODE ON
    FOLDER thirdparty)

# Do not propagate -Werror from the top-level project into upstream sources.
set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_WARNING_AS_ERROR OFF)

ov_install_static_lib(${TARGET_NAME} ${OV_CPACK_COMP_CORE})

ov_developer_package_export_targets(
    TARGET ${TARGET_NAME}
    INSTALL_INCLUDE_DIRECTORIES
        "${LIBURING_INCLUDE_DIR}/"
        "${LIBURING_GEN_INCLUDE_DIR}/")
