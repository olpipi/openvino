// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @file main.cpp
 * @brief Benchmarks file-reading backends: ifstream, mmap, and io_uring.
 *
 *   Backend      | API used
 *   -------------+----------------------------------------------------------
 *   ifstream     | ov::read_tensor_data (std::ifstream)
 *   mmap         | ov::read_tensor_data (mmap + parallel prefault)
 *   io_uring     | raw SYS_io_uring_setup / SYS_io_uring_enter syscalls,
 *                |   IORING_OP_READ, batched async submissions
 *
 * Drop page cache before each run for real SSD measurements:
 *   sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
 *
 * Build (within the OpenVINO source tree):
 *   cmake --build <build-dir> --target hello_classification
 *
 * Run:
 *   ./hello_classification [path_to_file]   (default: /tmp/read_bench_test.bin)
 */

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
// clang-format off
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
// clang-format on
#else
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <unistd.h>
#  ifdef __linux__
#    include <sys/syscall.h>
#    include <linux/io_uring.h>
#  endif
#endif

#include "openvino/runtime/tensor.hpp"

// ============================================================================
// io_uring raw implementation (no liburing dependency)  [Linux only]
// ============================================================================

namespace {

#ifdef __linux__

// Thin wrappers around io_uring syscalls
static int uring_setup(unsigned entries, struct io_uring_params* p) {
    return static_cast<int>(syscall(SYS_io_uring_setup, entries, p));
}
static int uring_enter(int fd, unsigned to_submit, unsigned min_complete, unsigned flags) {
    return static_cast<int>(syscall(SYS_io_uring_enter, fd, to_submit, min_complete, flags, nullptr, 0));
}

// Raw ring state (no liburing)
struct Ring {
    int fd = -1;

    // Submission queue pointers into the mmap'd region
    uint32_t* sq_head        = nullptr;
    uint32_t* sq_tail        = nullptr;
    uint32_t* sq_ring_mask   = nullptr;
    uint32_t* sq_ring_entries= nullptr;
    uint32_t* sq_array       = nullptr;
    io_uring_sqe* sqes       = nullptr;

    // Completion queue pointers
    uint32_t* cq_head        = nullptr;
    uint32_t* cq_tail        = nullptr;
    uint32_t* cq_ring_mask   = nullptr;
    io_uring_cqe* cqes       = nullptr;

    // mmap regions to unmap on destroy
    void*  sq_ring_ptr  = MAP_FAILED;  size_t sq_ring_size  = 0;
    void*  cq_ring_ptr  = MAP_FAILED;  size_t cq_ring_size  = 0;
    void*  sqes_ptr     = MAP_FAILED;  size_t sqes_size     = 0;
};

Ring ring_init(unsigned queue_depth) {
    io_uring_params params{};
    Ring r;
    r.fd = uring_setup(queue_depth, &params);
    if (r.fd < 0)
        throw std::runtime_error(std::string("io_uring_setup failed: ") + strerror(errno));

    // --- Map SQ ring ---
    r.sq_ring_size = params.sq_off.array + params.sq_entries * sizeof(uint32_t);
    r.sq_ring_ptr = mmap(nullptr, r.sq_ring_size,
                         PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE,
                         r.fd, IORING_OFF_SQ_RING);
    if (r.sq_ring_ptr == MAP_FAILED)
        throw std::runtime_error(std::string("mmap SQ ring failed: ") + strerror(errno));

    uint8_t* sq_base = static_cast<uint8_t*>(r.sq_ring_ptr);
    r.sq_head         = reinterpret_cast<uint32_t*>(sq_base + params.sq_off.head);
    r.sq_tail         = reinterpret_cast<uint32_t*>(sq_base + params.sq_off.tail);
    r.sq_ring_mask    = reinterpret_cast<uint32_t*>(sq_base + params.sq_off.ring_mask);
    r.sq_ring_entries = reinterpret_cast<uint32_t*>(sq_base + params.sq_off.ring_entries);
    r.sq_array        = reinterpret_cast<uint32_t*>(sq_base + params.sq_off.array);

    // --- Map CQ ring ---
    // With IORING_FEAT_SINGLE_MMAP both rings share one mmap; otherwise map separately.
    if (params.features & IORING_FEAT_SINGLE_MMAP) {
        r.cq_ring_ptr  = r.sq_ring_ptr;
        r.cq_ring_size = 0;  // nothing to unmap separately
    } else {
        r.cq_ring_size = params.cq_off.cqes + params.cq_entries * sizeof(io_uring_cqe);
        r.cq_ring_ptr  = mmap(nullptr, r.cq_ring_size,
                              PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE,
                              r.fd, IORING_OFF_CQ_RING);
        if (r.cq_ring_ptr == MAP_FAILED)
            throw std::runtime_error(std::string("mmap CQ ring failed: ") + strerror(errno));
    }

    uint8_t* cq_base = static_cast<uint8_t*>(r.cq_ring_ptr);
    r.cq_head      = reinterpret_cast<uint32_t*>(cq_base + params.cq_off.head);
    r.cq_tail      = reinterpret_cast<uint32_t*>(cq_base + params.cq_off.tail);
    r.cq_ring_mask = reinterpret_cast<uint32_t*>(cq_base + params.cq_off.ring_mask);
    r.cqes         = reinterpret_cast<io_uring_cqe*>(cq_base + params.cq_off.cqes);

    // --- Map SQE array ---
    r.sqes_size = params.sq_entries * sizeof(io_uring_sqe);
    r.sqes_ptr  = mmap(nullptr, r.sqes_size,
                       PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE,
                       r.fd, IORING_OFF_SQES);
    if (r.sqes_ptr == MAP_FAILED)
        throw std::runtime_error(std::string("mmap SQEs failed: ") + strerror(errno));
    r.sqes = static_cast<io_uring_sqe*>(r.sqes_ptr);

    return r;
}

void ring_destroy(Ring& r) {
    if (r.sqes_ptr != MAP_FAILED)    { munmap(r.sqes_ptr, r.sqes_size);       r.sqes_ptr = MAP_FAILED; }
    if (r.cq_ring_size > 0 && r.cq_ring_ptr != MAP_FAILED)
                                     { munmap(r.cq_ring_ptr, r.cq_ring_size); r.cq_ring_ptr = MAP_FAILED; }
    if (r.sq_ring_ptr != MAP_FAILED) { munmap(r.sq_ring_ptr, r.sq_ring_size); r.sq_ring_ptr = MAP_FAILED; }
    if (r.fd >= 0)                   { close(r.fd);                           r.fd = -1; }
}

/**
 * Sliding-window io_uring read.
 *
 * 1. Fill the ring to `depth` and submit all at once.
 * 2. Wait for at least one CQE.
 * 3. For every CQE reaped, immediately enqueue the next pending chunk.
 * 4. Submit all new SQEs in one syscall, go to 2.
 *
 * The kernel's I/O pipeline is never starved between rounds.
 */
void uring_read_file(Ring& r, int fd, char* dst, size_t size, size_t offset,
                     size_t chunk_size) {
    const uint32_t depth   = *r.sq_ring_entries;
    const uint32_t sq_mask = *r.sq_ring_mask;
    const uint32_t cq_mask = *r.cq_ring_mask;

    const size_t total_chunks = (size + chunk_size - 1) / chunk_size;
    size_t submitted = 0;
    size_t completed = 0;

    // Enqueue one SQE for chunk index chunk_idx.
    auto enqueue_sqe = [&](size_t chunk_idx) {
        const size_t byte_off   = chunk_idx * chunk_size;
        const size_t this_chunk = std::min(chunk_size, size - byte_off);

        const uint32_t tail = *r.sq_tail;
        const uint32_t slot = tail & sq_mask;

        io_uring_sqe& sqe = r.sqes[slot];
        sqe           = {};
        sqe.opcode    = IORING_OP_READ;
        sqe.fd        = fd;
        sqe.off       = static_cast<uint64_t>(offset + byte_off);
        sqe.addr      = reinterpret_cast<uint64_t>(dst + byte_off);
        sqe.len       = static_cast<uint32_t>(this_chunk);
        sqe.user_data = static_cast<uint64_t>(chunk_idx);

        r.sq_array[slot] = slot;
        __atomic_store_n(r.sq_tail, tail + 1, __ATOMIC_RELEASE);
    };

    // Phase 1: fill ring to capacity and submit in one syscall.
    const uint32_t initial = static_cast<uint32_t>(std::min(static_cast<size_t>(depth), total_chunks));
    for (uint32_t i = 0; i < initial; ++i)
        enqueue_sqe(submitted++);

    int ret = uring_enter(r.fd, initial, 0, 0);
    if (ret < 0)
        throw std::runtime_error(std::string("io_uring_enter initial submit: ") + strerror(errno));

    uint32_t in_flight = initial;

    // Phase 2: sliding window – restock one slot per CQE consumed.
    while (completed < total_chunks) {
        // Block until at least one completion arrives.
        ret = uring_enter(r.fd, 0, 1, IORING_ENTER_GETEVENTS);
        if (ret < 0)
            throw std::runtime_error(std::string("io_uring_enter wait: ") + strerror(errno));

        uint32_t cq_head       = __atomic_load_n(r.cq_head, __ATOMIC_ACQUIRE);
        const uint32_t cq_tail = __atomic_load_n(r.cq_tail, __ATOMIC_ACQUIRE);
        uint32_t new_subs = 0;

        while (cq_head != cq_tail) {
            const io_uring_cqe& cqe = r.cqes[cq_head & cq_mask];
            if (cqe.res < 0)
                throw std::runtime_error(std::string("io_uring read error: ") + strerror(-cqe.res));
            ++cq_head;
            --in_flight;
            ++completed;

            // Immediately refill the freed slot.
            if (submitted < total_chunks) {
                enqueue_sqe(submitted++);
                ++in_flight;
                ++new_subs;
            }
        }
        __atomic_store_n(r.cq_head, cq_head, __ATOMIC_RELEASE);

        if (new_subs > 0) {
            ret = uring_enter(r.fd, new_subs, 0, 0);
            if (ret < 0)
                throw std::runtime_error(std::string("io_uring_enter restock: ") + strerror(errno));
        }
    }
}

/**
 * Parallel io_uring read.
 *
 * Spawns `num_threads` worker threads, each owning its own ring and fd.
 * The file is split into `num_threads` equal-sized stripes; every worker
 * runs `uring_read_file` on its own stripe directly into the destination
 * buffer.  Threads are joined before the function returns.
 */
void uring_read_file_parallel(const std::string& path,
                              char* dst,
                              size_t size,
                              size_t offset,
                              unsigned num_threads,
                              unsigned queue_depth_per_thread,
                              size_t chunk_size) {
    const size_t stripe = (size + num_threads - 1) / num_threads;

    std::vector<std::thread> workers;
    std::vector<std::string> errors(num_threads);
    workers.reserve(num_threads);

    for (unsigned t = 0; t < num_threads; ++t) {
        workers.emplace_back([&, t] {
            const size_t off   = offset + t * stripe;
            const size_t avail = (off < offset + size) ? (offset + size - off) : 0;
            if (avail == 0)
                return;
            const size_t len = std::min(stripe, avail);

            int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
            if (fd < 0) {
                errors[t] = std::string("open failed: ") + strerror(errno);
                return;
            }

            Ring r{};
            try {
                r = ring_init(queue_depth_per_thread);
                uring_read_file(r, fd, dst + (off - offset), len, off, chunk_size);
            } catch (const std::exception& ex) {
                errors[t] = ex.what();
            }
            ring_destroy(r);
            close(fd);
        });
    }

    for (auto& w : workers)
        w.join();

    for (unsigned t = 0; t < num_threads; ++t)
        if (!errors[t].empty())
            throw std::runtime_error("Thread " + std::to_string(t) + ": " + errors[t]);
}

#endif  // __linux__

// ============================================================================
// Benchmark helpers
// ============================================================================

using Clock = std::chrono::steady_clock;

static long long elapsed_ms(Clock::time_point from) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - from).count();
}

// Best-effort cold-cache eviction for a single file.
static void drop_page_cache(const std::string& path) {
#ifdef _WIN32
    // Windows has no public API to evict cached file pages.
    // Opening with FILE_FLAG_NO_BUFFERING bypasses the cache for this handle;
    // for a reliable cold-cache run use RAMMap (Sysinternals) to empty the
    // standby list before benchmarking.
    HANDLE h = ::CreateFileA(path.c_str(), GENERIC_READ,
                             FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                             OPEN_EXISTING,
                             FILE_FLAG_NO_BUFFERING | FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h != INVALID_HANDLE_VALUE)
        ::CloseHandle(h);
#else
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
#endif
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
#ifdef _WIN32
    ::DiscardVirtualMemory(dst.data(), sz);
#else
    madvise(dst.data(), sz, MADV_DONTNEED);
#endif
    return dst;
}

// Drop file page cache AND destination buffer pages, then time the operation.
// Macro-like helper to keep call sites short.
#ifdef _WIN32
#  define COLD_START(path_, dst_)                                  \
    drop_page_cache(path_);                                        \
    ::DiscardVirtualMemory((dst_).data(), (dst_).size());          \
    auto t0_ = Clock::now();
#else
#  define COLD_START(path_, dst_)                                  \
    drop_page_cache(path_);                                        \
    madvise((dst_).data(), (dst_).size(), MADV_DONTNEED);          \
    auto t0_ = Clock::now();
#endif

}  // namespace

// ============================================================================
// main
// ============================================================================

int main(int argc, char* argv[]) {
    const std::string dir = (argc > 1) ? argv[1] : "/tmp";

    // File sizes to sweep: 10 MB → 10 GB
    const std::vector<size_t> sizes = {
        10ULL   * 1024 * 1024,
        50ULL   * 1024 * 1024,
        100ULL  * 1024 * 1024,
        500ULL  * 1024 * 1024,
        1ULL    * 1024 * 1024 * 1024,
        2ULL    * 1024 * 1024 * 1024,
        5ULL    * 1024 * 1024 * 1024,
        10ULL   * 1024 * 1024 * 1024,
    };

    const unsigned hw_threads = std::max(1u, std::thread::hardware_concurrency());
    constexpr unsigned URING_Q_SINGLE   = 256;
    constexpr unsigned URING_Q_PARALLEL = 64;
    constexpr size_t   URING_CHUNK      = 1UL * 1024 * 1024;

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
    for (size_t sz : sizes) {
        const std::string path = dir + "/bench_" + std::to_string(sz >> 20) + "MB.bin";
        const std::string size_label = fmt_size(sz);

        // ------------------------------------------------------------------
        // 1. ifstream via read_tensor_data
        //    Reads into an internal tensor buffer, then memcpy to dst.
        //    dst is freshly allocated and evicted from RAM before timing.
        // ------------------------------------------------------------------
        {
            auto dst = make_cold_dst(sz);
            COLD_START(path, dst);
            const auto tensor = ov::read_tensor_data(path, ov::element::u8,
                                                     ov::PartialShape::dynamic(1), 0, false);
            const long long read_ms = elapsed_ms(t0_);
            auto t1 = Clock::now();
            memcpy(dst.data(), tensor.data(), sz);
            const long long copy_ms = elapsed_ms(t1);
            print_row((size_label + "  ifstream").c_str(),
                      read_ms + copy_ms, read_ms, copy_ms, sz);
        }

        // ------------------------------------------------------------------
        // 2. mmap + parallel prefault
        //    Maps the file and faults all pages in with threads, then memcpy.
        // ------------------------------------------------------------------
        {
            auto dst = make_cold_dst(sz);
            COLD_START(path, dst);
            const auto tensor = ov::read_tensor_data(path, ov::element::u8,
                                                     ov::PartialShape::dynamic(1), 0, true);
            const long long read_ms = elapsed_ms(t0_);
            auto t1 = Clock::now();
            memcpy(dst.data(), tensor.data(), sz);
            const long long copy_ms = elapsed_ms(t1);
            print_row((size_label + "  mmap+prefault").c_str(),
                      read_ms + copy_ms, read_ms, copy_ms, sz);
        }

        // // ------------------------------------------------------------------
        // // 3. io_uring single-threaded
        // //    Reads directly into dst – no intermediate buffer, no memcpy.
        // // ------------------------------------------------------------------
        // {
        //     auto dst = make_cold_dst(sz);
        //     COLD_START(path, dst);
        //     int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
        //     if (fd < 0)
        //         throw std::runtime_error(std::string("open: ") + strerror(errno));
        //     Ring r = ring_init(URING_Q_SINGLE);
        //     uring_read_file(r, fd, reinterpret_cast<char*>(dst.data()), sz, 0, URING_CHUNK);
        //     const long long read_ms = elapsed_ms(t0_);
        //     ring_destroy(r);
        //     close(fd);
        //     print_row((size_label + "  io_uring(1T)").c_str(),
        //               read_ms, read_ms, 0, sz);
        // }

        // // ------------------------------------------------------------------
        // // 4. io_uring parallel
        // //    N threads, each own ring+fd, stripes of the file, direct to dst.
        // // ------------------------------------------------------------------
        // {
        //     auto dst = make_cold_dst(sz);
        //     COLD_START(path, dst);
        //     uring_read_file_parallel(path,
        //                              reinterpret_cast<char*>(dst.data()),
        //                              sz, 0,
        //                              hw_threads,
        //                              URING_Q_PARALLEL,
        //                              URING_CHUNK);
        //     const long long read_ms = elapsed_ms(t0_);
        //     print_row((size_label + "  io_uring(" + std::to_string(hw_threads) + "T)").c_str(),
        //               read_ms, read_ms, 0, sz);
        // }

        std::cout << "\n";
    }

    return 0;
}
