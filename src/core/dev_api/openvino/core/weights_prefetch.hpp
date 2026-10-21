// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <future>
#include <memory>
#include <streambuf>
#include <unordered_map>
#include <vector>

#include "openvino/core/core_visibility.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/runtime/aligned_buffer.hpp"
#include "openvino/util/memory.hpp"

namespace ov::util {

/** @brief Loads the weights before they are read and evicts them after use. */
class OPENVINO_API IPrefetchScheduler {
public:
    virtual ~IPrefetchScheduler() = default;

    /** @brief Waits for the node weights to be loaded. Call it right before the node is read. */
    virtual void acquire(const ov::Node& node) noexcept = 0;

    /** @brief Evicts the node weights and prefetches the next ones. Call it once the node is not read. */
    virtual void release(const ov::Node& node) noexcept = 0;
};

/**
 * @brief Loads the constants ahead of their consumer and evicts them after use.
 *
 * Fits a plugin that reads each constant once, in the order of ops. With lookahead = 2 and constants A B C D:
 * prefetch A, B; acquire A, release A (evict A, prefetch C); acquire B, release B (evict B, prefetch D); ...
 *
 * Non-constant nodes and constants not backed by a file mapping are ignored. Not thread safe.
 */
class OPENVINO_API OrderedPrefetchScheduler : public IPrefetchScheduler {
public:
    /**
     * @param ops Nodes in the order they are consumed; only the constants among them are tracked,
     *            duplicates are ignored.
     * @param lookahead Minimum number of constants prefetched and not released yet.
     * @param min_bytes More constants are prefetched while fewer bytes than this are prefetched and not released yet.
     */
    explicit OrderedPrefetchScheduler(const std::vector<std::shared_ptr<ov::Node>>& ops,
                                      size_t lookahead = 2,
                                      size_t min_bytes = 100UL * 1024 * 1024);

    void acquire(const ov::Node& node) noexcept override;
    void release(const ov::Node& node) noexcept override;

private:
    enum class State { pending, prefetched, released };

    void prefetch_ahead() noexcept;

    std::vector<std::shared_ptr<ov::op::v0::Constant>> m_constants;
    std::vector<State> m_states;
    std::vector<std::shared_future<void>> m_futures;
    std::unordered_map<const ov::Node*, size_t> m_indices;
    size_t m_lookahead;
    size_t m_min_bytes;
    size_t m_next = 0;
    size_t m_in_flight = 0;
    size_t m_in_flight_bytes = 0;
};

/**
 * @brief Prefetches a memory range read front to back in a window ahead of the reader and evicts what was read.
 *
 * Hints go to the buffer owning the range, so they take effect only for file mappings. Not thread safe.
 */
class OPENVINO_API SequentialPrefetchScheduler {
public:
    /**
     * @param source Buffer owning the memory, [data, data + size) must lie inside of it.
     * @param window Number of bytes kept prefetched ahead of the read position.
     * @param step Granularity of the hints; small hints are ignored by the file mappings.
     */
    SequentialPrefetchScheduler(std::shared_ptr<ov::AlignedBuffer> source,
                                const void* data,
                                size_t size,
                                size_t window = 100UL * 1024 * 1024,
                                size_t step = 16UL * 1024 * 1024);

    /** @brief Reports that [begin, end) is read next and the bytes before @p begin are consumed. */
    void advance(size_t begin, size_t end) noexcept;

    /** @brief Stops evicting, for when the read memory stays referenced (e.g. zero-copy import). */
    void keep_read_data() noexcept {
        m_evict = false;
    }

private:
    std::shared_ptr<ov::AlignedBuffer> make_chunk(size_t begin, size_t end) const;

    std::shared_ptr<ov::AlignedBuffer> m_source;
    char* m_data;
    size_t m_size;
    size_t m_window;
    size_t m_step;
    size_t m_prefetched = 0;
    size_t m_evicted = 0;
    bool m_evict = true;
};

/**
 * @brief Read-only stream buffer over memory that drives a SequentialPrefetchScheduler when the memory is owned by
 * a known buffer, and is a plain memory reader otherwise.
 */
class OPENVINO_API PrefetchStreamBuf : public std::streambuf {
public:
    PrefetchStreamBuf(const void* data, size_t size, std::shared_ptr<ov::AlignedBuffer> source);

    void keep_read_data() noexcept {
        if (m_scheduler) {
            m_scheduler->keep_read_data();
        }
    }

protected:
    std::streamsize xsgetn(char_type* dst, std::streamsize count) override;
    pos_type seekoff(off_type off, std::ios_base::seekdir dir, std::ios_base::openmode which) override;
    pos_type seekpos(pos_type pos, std::ios_base::openmode which) override;

private:
    std::unique_ptr<SequentialPrefetchScheduler> m_scheduler;
};
}  // namespace ov::util
