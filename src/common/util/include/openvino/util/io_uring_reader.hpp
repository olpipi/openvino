// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Async ring-based parallel file reader (Linux io_uring / Windows IoRing).
 * @file io_uring_reader.hpp
 *
 * Provides a single-call helper that reads a contiguous byte range from a file
 * into a caller-supplied destination buffer using asynchronous ring-based I/O,
 * distributing the work across N worker threads, each owning its own ring and
 * file descriptor/handle.
 *
 * On Linux the implementation uses io_uring (via liburing). On Windows it uses
 * the IoRing API (available in Windows 11 Build 22000+, loaded dynamically so
 * the binary remains compatible with older Windows versions).
 *
 * On platforms where neither ring-based I/O backend is available,
 * @ref is_io_uring_available() returns false and @ref read_file_io_uring()
 * refuses the request (returns false). Callers are expected to provide an
 * alternative I/O path (mmap, pread, ifstream, ReadFile).
 */

#pragma once

#include <cstddef>
#include <filesystem>

namespace ov::util {

/**
 * @brief Return true if the running OS and the build configuration both
 *        support async ring-based read operations.
 *
 * On Linux the probe performs a one-time io_uring_queue_init /
 * io_uring_queue_exit and caches the result. On Windows it dynamically loads
 * the IoRing functions from KernelBase.dll and creates/closes a test ring.
 *
 * Returns false when:
 *   - Linux: built without liburing (ENABLE_LIBURING=OFF), or kernel < 5.1,
 *     or io_uring administratively disabled.
 *   - Windows: built without IoRing support (ENABLE_IO_RING=OFF), or running
 *     on Windows 10 or earlier where the IoRing APIs do not exist.
 */
bool is_io_uring_available();

/**
 * @brief Read @p size bytes from @p path starting at byte @p file_offset into
 *        @p dst using parallel async ring-based I/O.
 *
 * The destination buffer must hold at least @p size bytes. The file is sliced
 * into roughly equal stripes; each worker thread opens its own file
 * descriptor/handle and its own ring instance.
 *
 * @param path        File path to read from.
 * @param dst         Destination buffer (caller owns).
 * @param size        Number of bytes to read.
 * @param file_offset Absolute starting offset in the file.
 * @return true on success, false on any I/O or setup failure. On failure, the
 *         contents of @p dst are unspecified.
 */
bool read_file_io_uring(const std::filesystem::path& path,
                        char* dst,
                        size_t size,
                        size_t file_offset);

}  // namespace ov::util
