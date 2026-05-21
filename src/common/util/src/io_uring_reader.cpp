// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/util/io_uring_reader.hpp"

#ifdef OPENVINO_ENABLE_LIBURING

#    include <fcntl.h>
#    include <liburing.h>
#    include <unistd.h>

#    include <algorithm>
#    include <atomic>
#    include <cstring>
#    include <mutex>
#    include <stdexcept>
#    include <string>
#    include <thread>
#    include <vector>

#    include "openvino/util/file_util.hpp"

namespace ov::util {

namespace {

// Tunables. Kept conservative so prototypes work well on cloud VMs that may
// limit the per-process pinned-page budget for io_uring rings.
constexpr unsigned URING_QUEUE_DEPTH = 256;
constexpr size_t URING_CHUNK_SIZE = 10UL * 1024 * 1024;  // 10 MiB per SQE
constexpr size_t URING_MIN_STRIPE = 8UL * 1024 * 1024;   // do not split below 8 MiB per worker

/**
 * Probe the kernel once for io_uring support. Caches the result so that hot
 * paths can call is_io_uring_available() cheaply.
 */
bool probe_io_uring_once() {
    io_uring ring{};
    const int ret = io_uring_queue_init(8, &ring, 0);
    if (ret != 0) {
        return false;
    }
    io_uring_queue_exit(&ring);
    return true;
}

/**
 * Sliding-window io_uring read for one stripe.
 *
 * Submits up to @c queue_depth chunks at a time and refills the freed slots as
 * completions come back, so the kernel pipeline is never idle between
 * iterations.
 */
bool uring_read_stripe(io_uring& ring,
                       int fd,
                       char* dst,
                       size_t size,
                       size_t file_offset,
                       unsigned queue_depth,
                       size_t chunk_size) {
    const size_t total_chunks = (size + chunk_size - 1) / chunk_size;
    if (total_chunks == 0) {
        return true;
    }

    auto submit_chunk = [&](size_t chunk_idx) -> bool {
        const size_t byte_off = chunk_idx * chunk_size;
        const size_t this_chunk = (std::min)(chunk_size, size - byte_off);
        io_uring_sqe* sqe = io_uring_get_sqe(&ring);
        if (!sqe) {
            return false;
        }
        io_uring_prep_read(sqe, fd, dst + byte_off, static_cast<unsigned>(this_chunk), file_offset + byte_off);
        io_uring_sqe_set_data64(sqe, static_cast<uint64_t>(chunk_idx));
        return true;
    };

    size_t submitted = 0;
    size_t completed = 0;
    unsigned in_flight = 0;

    const unsigned initial = static_cast<unsigned>((std::min)(static_cast<size_t>(queue_depth), total_chunks));
    for (unsigned i = 0; i < initial; ++i) {
        if (!submit_chunk(submitted++)) {
            return false;
        }
        ++in_flight;
    }
    if (io_uring_submit(&ring) < 0) {
        return false;
    }

    while (completed < total_chunks) {
        io_uring_cqe* cqe = nullptr;
        int wait_ret = io_uring_wait_cqe(&ring, &cqe);
        if (wait_ret < 0 || !cqe) {
            return false;
        }

        // Drain everything currently in the CQ in one pass, ack with a single cq_advance.
        bool error_seen = false;
        unsigned drained = 0;
        unsigned head = 0;
        io_uring_cqe* it_cqe = nullptr;
        io_uring_for_each_cqe(&ring, head, it_cqe) {
            const int res = it_cqe->res;
            const uint64_t idx = io_uring_cqe_get_data64(it_cqe);
            const size_t expected =
                (std::min)(chunk_size, size - static_cast<size_t>(idx) * chunk_size);
            if (res < 0 || static_cast<size_t>(res) < expected) {
                error_seen = true;
            }
            ++drained;
        }
        io_uring_cq_advance(&ring, drained);
        completed += drained;
        in_flight -= drained;

        if (error_seen) {
            return false;
        }

        // Restock freed slots.
        unsigned new_subs = 0;
        while (submitted < total_chunks && in_flight < queue_depth) {
            if (!submit_chunk(submitted++)) {
                break;
            }
            ++in_flight;
            ++new_subs;
        }
        if (new_subs > 0) {
            if (io_uring_submit(&ring) < 0) {
                return false;
            }
        }
    }
    return true;
}

}  // namespace

bool is_io_uring_available() {
    static const bool cached = probe_io_uring_once();
    return cached;
}

bool read_file_io_uring(const std::filesystem::path& path, char* dst, size_t size, size_t file_offset) {
    if (size == 0) {
        return true;
    }
    if (!dst || !is_io_uring_available()) {
        return false;
    }

    // Pick a thread count proportional to the read size; cap by hw concurrency.
    const size_t hw = (std::max)(size_t{1}, static_cast<size_t>(std::thread::hardware_concurrency()));
    const size_t by_size = (std::max)(size_t{1}, size / URING_MIN_STRIPE);
    const size_t num_threads = (std::min)(hw, by_size);

    const size_t stripe = (size + num_threads - 1) / num_threads;

    std::atomic<bool> ok{true};
    std::vector<std::thread> workers;
    workers.reserve(num_threads);

    for (size_t t = 0; t < num_threads; ++t) {
        workers.emplace_back([&, t]() {
            const size_t off = t * stripe;
            if (off >= size) {
                return;
            }
            const size_t len = (std::min)(stripe, size - off);
            const std::string path_str = ov::util::path_to_string(path);
            const int fd = ::open(path_str.c_str(), O_RDONLY | O_CLOEXEC);
            if (fd < 0) {
                ok.store(false, std::memory_order_relaxed);
                return;
            }
            io_uring ring{};
            if (io_uring_queue_init(URING_QUEUE_DEPTH, &ring, 0) != 0) {
                ::close(fd);
                ok.store(false, std::memory_order_relaxed);
                return;
            }
            const bool ret = uring_read_stripe(ring,
                                               fd,
                                               dst + off,
                                               len,
                                               file_offset + off,
                                               URING_QUEUE_DEPTH,
                                               URING_CHUNK_SIZE);
            io_uring_queue_exit(&ring);
            ::close(fd);
            if (!ret) {
                ok.store(false, std::memory_order_relaxed);
            }
        });
    }
    for (auto& w : workers) {
        w.join();
    }
    return ok.load(std::memory_order_relaxed);
}

}  // namespace ov::util

#elif defined(OPENVINO_ENABLE_WIN_IO_RING)

#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    include <windows.h>

#    include <algorithm>
#    include <atomic>
#    include <cstring>
#    include <mutex>
#    include <string>
#    include <thread>
#    include <vector>

#    include "openvino/util/file_util.hpp"

namespace ov::util {

namespace {

// ============================================================================
// Windows IoRing type definitions (ABI-compatible with ioringapi.h from
// Windows SDK 10.0.22000+). Defined here to avoid hard SDK version requirement.
// ============================================================================

enum IORING_VERSION_T : UINT32 {
    OV_IORING_VERSION_INVALID = 0,
    OV_IORING_VERSION_1 = 1,
    OV_IORING_VERSION_2 = 2,
    OV_IORING_VERSION_3 = 3,
};

enum IORING_CREATE_REQUIRED_FLAGS_T : UINT32 {
    OV_IORING_CREATE_REQUIRED_FLAGS_NONE = 0,
};

enum IORING_CREATE_ADVISORY_FLAGS_T : UINT32 {
    OV_IORING_CREATE_ADVISORY_FLAGS_NONE = 0,
};

struct IORING_CREATE_FLAGS_T {
    IORING_CREATE_REQUIRED_FLAGS_T Required;
    IORING_CREATE_ADVISORY_FLAGS_T Advisory;
};

enum IORING_REF_KIND_T : UINT32 {
    OV_IORING_REF_RAW = 0,
    OV_IORING_REF_REGISTERED = 1,
};

enum IORING_SQE_FLAGS_T : UINT32 {
    OV_IOSQE_FLAGS_NONE = 0,
};

struct IORING_HANDLE_REF_T {
    IORING_REF_KIND_T Kind;
    union {
        HANDLE Handle;
        UINT32 Index;
    };
};

struct IORING_BUFFER_REF_T {
    IORING_REF_KIND_T Kind;
    union {
        void* Address;
        struct {
            UINT32 BufferIndex;
            UINT32 Offset;
        } IndexAndOffset;
    };
};

struct IORING_CQE_T {
    UINT_PTR UserData;
    HRESULT ResultCode;
    ULONG_PTR Information;
};

struct IORING_CAPABILITIES_T {
    IORING_VERSION_T MaxVersion;
    UINT32 MaxSubmissionQueueSize;
    UINT32 MaxCompletionQueueSize;
    UINT32 FeatureFlags;
};

// Opaque handle type (matches HIORING from SDK).
DECLARE_HANDLE(HIORING_T);

// ============================================================================
// Function pointer types for dynamic loading.
// ============================================================================

using PFN_QueryIoRingCapabilities = HRESULT(WINAPI*)(IORING_CAPABILITIES_T*);
using PFN_CreateIoRing = HRESULT(WINAPI*)(IORING_VERSION_T, IORING_CREATE_FLAGS_T, UINT32, UINT32, HIORING_T**);
using PFN_CloseIoRing = HRESULT(WINAPI*)(HIORING_T*);
using PFN_BuildIoRingReadFile = HRESULT(WINAPI*)(HIORING_T*,
                                                  IORING_HANDLE_REF_T,
                                                  IORING_BUFFER_REF_T,
                                                  UINT32,
                                                  UINT64,
                                                  UINT_PTR,
                                                  IORING_SQE_FLAGS_T);
using PFN_SubmitIoRing = HRESULT(WINAPI*)(HIORING_T*, UINT32, UINT32, UINT32*);
using PFN_PopIoRingCompletion = HRESULT(WINAPI*)(HIORING_T*, IORING_CQE_T*);

// ============================================================================
// Dynamically resolved function pointers (loaded once).
// ============================================================================

struct IoRingFunctions {
    PFN_QueryIoRingCapabilities pQueryIoRingCapabilities = nullptr;
    PFN_CreateIoRing pCreateIoRing = nullptr;
    PFN_CloseIoRing pCloseIoRing = nullptr;
    PFN_BuildIoRingReadFile pBuildIoRingReadFile = nullptr;
    PFN_SubmitIoRing pSubmitIoRing = nullptr;
    PFN_PopIoRingCompletion pPopIoRingCompletion = nullptr;

    bool valid() const {
        return pQueryIoRingCapabilities && pCreateIoRing && pCloseIoRing &&
               pBuildIoRingReadFile && pSubmitIoRing && pPopIoRingCompletion;
    }
};

IoRingFunctions load_io_ring_functions() {
    IoRingFunctions fns{};
    HMODULE hMod = ::GetModuleHandleW(L"KernelBase.dll");
    if (!hMod) {
        hMod = ::GetModuleHandleW(L"kernel32.dll");
    }
    if (!hMod) {
        return fns;
    }
    fns.pQueryIoRingCapabilities =
        reinterpret_cast<PFN_QueryIoRingCapabilities>(::GetProcAddress(hMod, "QueryIoRingCapabilities"));
    fns.pCreateIoRing = reinterpret_cast<PFN_CreateIoRing>(::GetProcAddress(hMod, "CreateIoRing"));
    fns.pCloseIoRing = reinterpret_cast<PFN_CloseIoRing>(::GetProcAddress(hMod, "CloseIoRing"));
    fns.pBuildIoRingReadFile =
        reinterpret_cast<PFN_BuildIoRingReadFile>(::GetProcAddress(hMod, "BuildIoRingReadFile"));
    fns.pSubmitIoRing = reinterpret_cast<PFN_SubmitIoRing>(::GetProcAddress(hMod, "SubmitIoRing"));
    fns.pPopIoRingCompletion =
        reinterpret_cast<PFN_PopIoRingCompletion>(::GetProcAddress(hMod, "PopIoRingCompletion"));
    return fns;
}

const IoRingFunctions& get_io_ring_fns() {
    static const IoRingFunctions fns = load_io_ring_functions();
    return fns;
}

// ============================================================================
// Tunables (mirroring the Linux io_uring implementation).
// ============================================================================

constexpr unsigned IORING_QUEUE_DEPTH = 256;
constexpr size_t IORING_CHUNK_SIZE = 10UL * 1024 * 1024;   // 10 MiB per SQE
constexpr size_t IORING_MIN_STRIPE = 8UL * 1024 * 1024;    // do not split below 8 MiB per worker

/**
 * Probe the OS once for IoRing support. Creates a small ring and immediately
 * closes it to verify runtime availability.
 */
bool probe_io_ring_once() {
    const auto& fns = get_io_ring_fns();
    if (!fns.valid()) {
        return false;
    }
    // Verify the kernel actually supports IoRing by creating a tiny ring.
    IORING_CREATE_FLAGS_T flags{};
    flags.Required = OV_IORING_CREATE_REQUIRED_FLAGS_NONE;
    flags.Advisory = OV_IORING_CREATE_ADVISORY_FLAGS_NONE;
    HIORING_T* ring = nullptr;
    HRESULT hr = fns.pCreateIoRing(OV_IORING_VERSION_1, flags, 8, 16, &ring);
    if (FAILED(hr) || !ring) {
        return false;
    }
    fns.pCloseIoRing(ring);
    return true;
}

/**
 * Batch-based IoRing read for one stripe.
 *
 * Submits up to @c queue_depth chunks at a time, waits for the whole batch to
 * complete, verifies results, then repeats until all data is read.
 */
bool ioring_read_stripe(const IoRingFunctions& fns,
                        HANDLE hFile,
                        char* dst,
                        size_t size,
                        size_t file_offset,
                        unsigned queue_depth,
                        size_t chunk_size) {
    const size_t total_chunks = (size + chunk_size - 1) / chunk_size;
    if (total_chunks == 0) {
        return true;
    }

    IORING_CREATE_FLAGS_T flags{};
    flags.Required = OV_IORING_CREATE_REQUIRED_FLAGS_NONE;
    flags.Advisory = OV_IORING_CREATE_ADVISORY_FLAGS_NONE;
    HIORING_T* ring = nullptr;
    HRESULT hr = fns.pCreateIoRing(OV_IORING_VERSION_1, flags, queue_depth, queue_depth * 2, &ring);
    if (FAILED(hr) || !ring) {
        return false;
    }

    IORING_HANDLE_REF_T fileRef{};
    fileRef.Kind = OV_IORING_REF_RAW;
    fileRef.Handle = hFile;

    size_t chunks_submitted = 0;
    size_t chunks_completed = 0;

    while (chunks_completed < total_chunks) {
        // Build a batch of read operations.
        const unsigned batch = static_cast<unsigned>(
            (std::min)(static_cast<size_t>(queue_depth), total_chunks - chunks_submitted));

        for (unsigned i = 0; i < batch; ++i) {
            const size_t chunk_idx = chunks_submitted + i;
            const size_t byte_off = chunk_idx * chunk_size;
            const size_t this_chunk = (std::min)(chunk_size, size - byte_off);

            IORING_BUFFER_REF_T bufRef{};
            bufRef.Kind = OV_IORING_REF_RAW;
            bufRef.Address = dst + byte_off;

            hr = fns.pBuildIoRingReadFile(ring,
                                          fileRef,
                                          bufRef,
                                          static_cast<UINT32>(this_chunk),
                                          static_cast<UINT64>(file_offset + byte_off),
                                          static_cast<UINT_PTR>(chunk_idx),
                                          OV_IOSQE_FLAGS_NONE);
            if (FAILED(hr)) {
                fns.pCloseIoRing(ring);
                return false;
            }
        }

        // Submit and wait for the full batch.
        UINT32 submitted = 0;
        hr = fns.pSubmitIoRing(ring, batch, INFINITE, &submitted);
        if (FAILED(hr)) {
            fns.pCloseIoRing(ring);
            return false;
        }

        // Pop and verify all completions from this batch.
        for (unsigned i = 0; i < batch; ++i) {
            IORING_CQE_T cqe{};
            hr = fns.pPopIoRingCompletion(ring, &cqe);
            if (hr != S_OK) {
                // S_FALSE means queue empty (unexpected), other = error.
                fns.pCloseIoRing(ring);
                return false;
            }
            if (FAILED(cqe.ResultCode)) {
                fns.pCloseIoRing(ring);
                return false;
            }
            // Verify the expected byte count was read.
            const size_t idx = static_cast<size_t>(cqe.UserData);
            const size_t expected = (std::min)(chunk_size, size - idx * chunk_size);
            if (cqe.Information < expected) {
                fns.pCloseIoRing(ring);
                return false;
            }
        }

        chunks_submitted += batch;
        chunks_completed += batch;
    }

    fns.pCloseIoRing(ring);
    return true;
}

}  // namespace

bool is_io_uring_available() {
    static const bool cached = probe_io_ring_once();
    return cached;
}

bool read_file_io_uring(const std::filesystem::path& path, char* dst, size_t size, size_t file_offset) {
    if (size == 0) {
        return true;
    }
    if (!dst || !is_io_uring_available()) {
        return false;
    }

    const auto& fns = get_io_ring_fns();

    // Pick a thread count proportional to the read size; cap by hw concurrency.
    const size_t hw = (std::max)(size_t{1}, static_cast<size_t>(std::thread::hardware_concurrency()));
    const size_t by_size = (std::max)(size_t{1}, size / IORING_MIN_STRIPE);
    const size_t num_threads = (std::min)(hw, by_size);

    const size_t stripe = (size + num_threads - 1) / num_threads;

    std::atomic<bool> ok{true};
    std::vector<std::thread> workers;
    workers.reserve(num_threads);

    for (size_t t = 0; t < num_threads; ++t) {
        workers.emplace_back([&, t]() {
            const size_t off = t * stripe;
            if (off >= size) {
                return;
            }
            const size_t len = (std::min)(stripe, size - off);

            // Open the file with overlapped flag for async I/O via IoRing.
            const HANDLE hFile = ::CreateFileW(path.c_str(),
                                               GENERIC_READ,
                                               FILE_SHARE_READ,
                                               nullptr,
                                               OPEN_EXISTING,
                                               FILE_FLAG_OVERLAPPED | FILE_FLAG_SEQUENTIAL_SCAN,
                                               nullptr);
            if (hFile == INVALID_HANDLE_VALUE) {
                ok.store(false, std::memory_order_relaxed);
                return;
            }

            const bool ret = ioring_read_stripe(fns,
                                                hFile,
                                                dst + off,
                                                len,
                                                file_offset + off,
                                                IORING_QUEUE_DEPTH,
                                                IORING_CHUNK_SIZE);
            ::CloseHandle(hFile);
            if (!ret) {
                ok.store(false, std::memory_order_relaxed);
            }
        });
    }
    for (auto& w : workers) {
        w.join();
    }
    return ok.load(std::memory_order_relaxed);
}

}  // namespace ov::util

#else  // !OPENVINO_ENABLE_LIBURING && !OPENVINO_ENABLE_WIN_IO_RING

namespace ov::util {

bool is_io_uring_available() {
    return false;
}

bool read_file_io_uring(const std::filesystem::path&, char*, size_t, size_t) {
    return false;
}

}  // namespace ov::util

#endif  // OPENVINO_ENABLE_LIBURING / OPENVINO_ENABLE_WIN_IO_RING
