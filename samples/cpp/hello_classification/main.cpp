// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

// clang-format off
#include "openvino/openvino.hpp"

#include "samples/args_helper.hpp"
#include "samples/common.hpp"
#include "samples/classification_results.h"
#include "samples/slog.hpp"
#include "format_reader_ptr.h"
// clang-format on

// ============================================================================
// Benchmark helpers
// ============================================================================

using Clock = std::chrono::steady_clock;

static long long elapsed_ms(Clock::time_point from) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - from).count();
}

// Best-effort cold-cache eviction for a single file.
static void drop_page_cache(const std::string& path) {
    // 1. posix_fadvise – asks the kernel to release cached pages for this file.
    int fd = open(path.c_str(), O_RDONLY);
    if (fd >= 0) {
        posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
        close(fd);
    }
    // 2. If we have write access, also write to the global drop_caches knob.
    //    For a guaranteed drop also run:
    //      sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
    {
        int kfd = open("/proc/sys/vm/drop_caches", O_WRONLY);
        if (kfd >= 0) {
            sync();
            const char val[] = "3";
            write(kfd, val, 1);
            close(kfd);
        }
    }
}

static std::string fmt_size(size_t bytes) {
    if (bytes >= (1ULL << 30)) {
        char buf[32];
        snprintf(buf, sizeof(buf), "%.0f GB", static_cast<double>(bytes) / (1ULL << 30));
        return buf;
    }
    char buf[32];
    snprintf(buf, sizeof(buf), "%zu MB", bytes >> 20);
    return buf;
}

// Create a file of `size` bytes filled with a deterministic pattern.
// Uses 8 MB streaming chunks so heap allocation is bounded regardless of size.
static void create_test_file(const std::string& path, size_t size) {
    std::cout << "  Creating " << path << " (" << fmt_size(size) << ") ... " << std::flush;
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f)
        throw std::runtime_error("Cannot create " + path);
    constexpr size_t CHUNK = 8UL * 1024 * 1024;
    std::vector<uint8_t> buf(std::min(CHUNK, size));
    size_t written = 0;
    while (written < size) {
        const size_t n = std::min(CHUNK, size - written);
        for (size_t i = 0; i < n; ++i)
            buf[i] = static_cast<uint8_t>(((written + i) * 1103515245ULL + 12345ULL) & 0xFFULL);
        f.write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(n));
        if (!f)
            throw std::runtime_error("Write failed: " + path);
        written += n;
    }
    std::cout << "done\n";
}

static void print_row(const char* method, long long total_ms, long long read_ms,
                      long long copy_ms, size_t bytes) {
    auto speed = [&](long long ms) -> double {
        return ms > 0 ? (static_cast<double>(bytes) / (ms / 1000.0)) / (1024.0 * 1024.0) : 0.0;
    };
    std::cout << "  " << std::left << std::setw(30) << method
              << std::right
              << std::setw(7) << read_ms << " ms read"
              << std::setw(7) << copy_ms << " ms cpy"
              << std::setw(7) << total_ms << " ms tot"
              << "   " << std::fixed << std::setprecision(0)
              << std::setw(6) << speed(total_ms) << " MB/s\n";
}

// Allocate a fresh destination buffer and immediately evict its pages from RAM
// so that writing into it during the timed section hits cold DRAM, not warm cache.
static std::vector<uint8_t> make_cold_dst(size_t sz) {
    std::vector<uint8_t> dst(sz);
    // Touch every page to trigger allocation, then evict from RAM.
    madvise(dst.data(), sz, MADV_DONTNEED);
    return dst;
}
#  define COLD_START(path_, dst_)                                  \
    drop_page_cache(path_);                                        \
    madvise((dst_).data(), (dst_).size(), MADV_DONTNEED);          \
    auto t0_ = Clock::now();




/**
 * @brief Main with support Unicode paths, wide strings
 */
int tmain(int argc, tchar* argv[]) {
    const std::string dir = (argc > 1) ? argv[1] : ".";

    // File sizes to sweep: 10 MB → 10 GB
    const std::vector<size_t> sizes = {
        // 1ULL   * 1024 * 1024,
        // 2ULL   * 1024 * 1024,
        5ULL   * 1024 * 1024,
        10ULL   * 1024 * 1024,
        20ULL   * 1024 * 1024,
        50ULL   * 1024 * 1024,
        100ULL  * 1024 * 1024,
        500ULL  * 1024 * 1024,
        1ULL    * 1024 * 1024 * 1024,
        2ULL    * 1024 * 1024 * 1024,
        5ULL    * 1024 * 1024 * 1024,
        10ULL   * 1024 * 1024 * 1024,
    };
    // const std::vector<size_t> sizes = {
    //     10ULL   * 1024 * 1024 * 1024,
    //     10ULL   * 1024 * 1024 * 1024,
    //     10ULL   * 1024 * 1024 * 1024,
    //     10ULL   * 1024 * 1024 * 1024,
    //     10ULL   * 1024 * 1024 * 1024,
    //     10ULL   * 1024 * 1024 * 1024,
    //     10ULL   * 1024 * 1024 * 1024,
    //     10ULL   * 1024 * 1024 * 1024,
    //     10ULL   * 1024 * 1024 * 1024,
    //     10ULL   * 1024 * 1024 * 1024,
    //     10ULL   * 1024 * 1024 * 1024,
    //     10ULL   * 1024 * 1024 * 1024,
    // };
    // -----------------------------------------------------------------------
    // Create test files (skipped if the file already exists with correct size)
    // -----------------------------------------------------------------------
    std::cout << "=== Preparing test files in " << dir << " ===\n";
    for (size_t sz : sizes) {
        const std::string path = dir + "/bench_" + std::to_string(sz >> 20) + "MB.bin";
        const bool exists = std::filesystem::exists(path) &&
                            std::filesystem::file_size(path) == sz;
        if (!exists)
            create_test_file(path, sz);
        else
            std::cout << "  " << path << " already exists, skipping.\n";
    }
    std::cout << "\n";

    // -----------------------------------------------------------------------
    // Header
    // -----------------------------------------------------------------------
    std::cout << std::left  << std::setw(8)  << "Size"
              << std::left  << std::setw(30) << "Method"
              << std::right << std::setw(14) << "Read"
              << std::setw(12) << "Memcpy"
              << std::setw(11) << "Total"
              << std::setw(12) << "Speed"
              << "\n" << std::string(87, '-') << "\n";

    // -----------------------------------------------------------------------
    // Sweep
    // -----------------------------------------------------------------------
    for (auto sz = sizes.rbegin(); sz != sizes.rend(); ++sz) {
        const std::string path = dir + "/bench_" + std::to_string(*sz >> 20) + "MB.bin";
        const std::string size_label = fmt_size(*sz);

        {
            auto dst = make_cold_dst(*sz);
            COLD_START(path, dst);
            const auto tensor = ov::read_tensor_data(path, ov::element::u8,
                                                     ov::PartialShape::dynamic(1), 0, false);
            const long long read_ms = elapsed_ms(t0_);
            auto t1 = Clock::now();
            memcpy(dst.data(), tensor.data(), *sz);
            const long long copy_ms = elapsed_ms(t1);
            print_row((size_label + "  read_tensor_data").c_str(),
                      read_ms + copy_ms, read_ms, copy_ms, *sz);
        }

        std::cout << "\n";
    }

    return EXIT_SUCCESS;
}
