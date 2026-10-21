// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/weights_prefetch.hpp"

#include <algorithm>
#include <cstring>
#include <tuple>
#include <utility>

#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/weight_sharing_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/runtime/shared_buffer.hpp"

namespace ov::util {

SequentialPrefetchScheduler::SequentialPrefetchScheduler(std::shared_ptr<ov::AlignedBuffer> source,
                                                         const void* data,
                                                         size_t size,
                                                         size_t window,
                                                         size_t step)
    : m_source(std::move(source)),
      m_data(static_cast<char*>(const_cast<void*>(data))),
      m_size(size),
      m_window(window),
      m_step(std::max<size_t>(step, 1)) {
    OPENVINO_ASSERT(m_source, "Prefetch source buffer is not set");
    const auto source_begin = static_cast<const char*>(std::as_const(*m_source).get_ptr());
    OPENVINO_ASSERT(m_data >= source_begin && m_size <= m_source->size() &&
                        static_cast<size_t>(m_data - source_begin) <= m_source->size() - m_size,
                    "Prefetch range is outside of the source buffer");
    advance(0, 0);
}

std::shared_ptr<ov::AlignedBuffer> SequentialPrefetchScheduler::make_chunk(size_t begin, size_t end) const {
    return std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::AlignedBuffer>>>(m_data + begin,
                                                                                  end - begin,
                                                                                  m_source);
}

void SequentialPrefetchScheduler::advance(size_t begin, size_t end) noexcept {
    try {
        begin = std::min(begin, m_size);
        end = std::min(std::max(begin, end), m_size);
        if (m_evict) {
            // Whole steps only, the read position may be in the middle of a page.
            const auto evict_end = begin == m_size ? m_size : begin / m_step * m_step;
            if (evict_end > m_evicted) {
                make_chunk(m_evicted, evict_end)->hint_evict();
                m_evicted = evict_end;
            }
        }
        m_prefetched = std::max(m_prefetched, begin);
        const auto target = std::min(end + m_window, m_size);
        while (m_prefetched < target) {
            const auto chunk_end = std::min(m_prefetched + m_step, m_size);
            std::ignore = make_chunk(m_prefetched, chunk_end)->hint_prefetch_async();
            m_prefetched = chunk_end;
        }
    } catch (...) {
        // Hints are advisory, the data is read correctly without them.
    }
}

PrefetchStreamBuf::PrefetchStreamBuf(const void* data, size_t size, std::shared_ptr<ov::AlignedBuffer> source) {
    const auto begin = static_cast<char*>(const_cast<void*>(data));
    setg(begin, begin, begin + size);
    if (source) {
        m_scheduler = std::make_unique<SequentialPrefetchScheduler>(std::move(source), data, size);
    }
}

std::streamsize PrefetchStreamBuf::xsgetn(char_type* dst, std::streamsize count) {
    const auto n = std::min<std::streamsize>(count, egptr() - gptr());
    if (n <= 0) {
        return 0;
    }
    const auto pos = static_cast<size_t>(gptr() - eback());
    if (m_scheduler) {
        m_scheduler->advance(pos, pos + static_cast<size_t>(n));
    }
    std::memcpy(dst, gptr(), static_cast<size_t>(n));
    // gbump() takes int, reads may exceed 2 GB.
    setg(eback(), gptr() + n, egptr());
    return n;
}

PrefetchStreamBuf::pos_type PrefetchStreamBuf::seekoff(off_type off,
                                                       std::ios_base::seekdir dir,
                                                       std::ios_base::openmode which) {
    if (!(which & std::ios_base::in)) {
        return pos_type(off_type(-1));
    }
    const off_type base = dir == std::ios_base::beg   ? 0
                          : dir == std::ios_base::cur ? gptr() - eback()
                                                      : egptr() - eback();
    const off_type pos = base + off;
    if (pos < 0 || pos > egptr() - eback()) {
        return pos_type(off_type(-1));
    }
    setg(eback(), eback() + pos, egptr());
    return pos_type(pos);
}

PrefetchStreamBuf::pos_type PrefetchStreamBuf::seekpos(pos_type pos, std::ios_base::openmode which) {
    return seekoff(off_type(pos), std::ios_base::beg, which);
}

OrderedPrefetchScheduler::OrderedPrefetchScheduler(const std::vector<std::shared_ptr<ov::Node>>& ops,
                                                   size_t lookahead,
                                                   size_t min_bytes)
    : m_lookahead(lookahead),
      m_min_bytes(min_bytes) {
    m_constants.reserve(ops.size());
    for (const auto& op : ops) {
        if (auto constant = ov::as_type_ptr<ov::op::v0::Constant>(op);
            constant && m_indices.emplace(constant.get(), m_constants.size()).second) {
            m_constants.push_back(std::move(constant));
        }
    }
    m_states.assign(m_constants.size(), State::pending);
    m_futures.resize(m_constants.size());
    prefetch_ahead();
}

void OrderedPrefetchScheduler::acquire(const ov::Node& node) noexcept {
    const auto found = m_indices.find(&node);
    if (found == m_indices.end()) {
        return;
    }
    const auto index = found->second;
    if (m_states[index] == State::pending) {
        // Consumed ahead of the planned order: still load it in the background, on several threads.
        m_futures[index] = ov::weight_sharing::Extension::hint_prefetch(*m_constants[index]);
        m_states[index] = State::prefetched;
        ++m_in_flight;
        m_in_flight_bytes += m_constants[index]->get_byte_size();
    }
}

void OrderedPrefetchScheduler::release(const ov::Node& node) noexcept {
    const auto found = m_indices.find(&node);
    if (found == m_indices.end() || m_states[found->second] == State::released) {
        return;
    }
    const auto index = found->second;
    if (m_states[index] == State::prefetched) {
        --m_in_flight;
        m_in_flight_bytes -= m_constants[index]->get_byte_size();
    }
    m_states[index] = State::released;
    m_futures[index] = {};
    ov::weight_sharing::Extension::hint_evict(*m_constants[index]);
    prefetch_ahead();
}

void OrderedPrefetchScheduler::prefetch_ahead() noexcept {
    for (; (m_in_flight < m_lookahead || m_in_flight_bytes < m_min_bytes) && m_next < m_constants.size(); ++m_next) {
        if (m_states[m_next] == State::pending) {
            m_futures[m_next] = ov::weight_sharing::Extension::hint_prefetch(*m_constants[m_next]);
            m_states[m_next] = State::prefetched;
            ++m_in_flight;
            m_in_flight_bytes += m_constants[m_next]->get_byte_size();
        }
    }
}

}  // namespace ov::util
