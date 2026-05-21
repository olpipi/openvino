// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief liburing-based parallel file reader.
 * @file io_uring_reader.hpp
 *
 * Provides a single-call helper that reads a contiguous byte range from a file
 * into a caller-supplied destination buffer using io_uring submission/completion
 * queues, distributing the work across N worker threads, each owning its own
 * ring and file descriptor.
 *
 * The implementation is Linux-only. On platforms where the library is built
 * without liburing support, @ref is_io_uring_available() returns false and
 * @ref read_file_io_uring() refuses the request (returns false). Callers are
 * expected to provide an alternative I/O path (mmap, pread, ifstream).
 */

#pragma once

#include <cstddef>
#include <filesystem>

namespace ov::util {

/**
 * @brief Return true if the running kernel and the build configuration both
 *        support io_uring read operations.
 *
 * The probe performs a one-time io_uring_queue_init / io_uring_queue_exit and
 * caches the result. It returns false when:
 *   - the build was configured without liburing (ENABLE_LIBURING=OFF), or
 *   - the runtime kernel does not implement io_uring (pre-5.1), or
 *   - io_uring is administratively disabled (e.g. sysctl kernel.io_uring_disabled).
 */
bool is_io_uring_available();

/**
 * @brief Read @p size bytes from @p path starting at byte @p file_offset into
 *        @p dst using parallel io_uring rings.
 *
 * The destination buffer must hold at least @p size bytes. The file is sliced
 * into roughly equal stripes; each worker thread opens its own file descriptor
 * (so per-fd readahead state stays independent) and its own io_uring instance.
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
