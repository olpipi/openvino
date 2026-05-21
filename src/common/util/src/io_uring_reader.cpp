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

#else  // !OPENVINO_ENABLE_LIBURING

namespace ov::util {

bool is_io_uring_available() {
    return false;
}

bool read_file_io_uring(const std::filesystem::path&, char*, size_t, size_t) {
    return false;
}

}  // namespace ov::util

#endif  // OPENVINO_ENABLE_LIBURING
