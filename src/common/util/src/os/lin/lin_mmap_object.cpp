// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <future>
#include <mutex>
#include <optional>
#include <thread>
#include <tuple>
#include <vector>

#include "memory_prefetch.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/hash_util.hpp"
#include "openvino/util/math_util.hpp"
#include "openvino/util/memory.hpp"
#include "openvino/util/mmap_object.hpp"
#include "openvino/util/parallel_io.hpp"

namespace ov {
namespace util {
int64_t get_system_page_size() {
    static auto page_size = static_cast<int64_t>(sysconf(_SC_PAGE_SIZE));
    return page_size;
}

size_t get_system_alloc_granularity() {
    return static_cast<size_t>(get_system_page_size());
}

/**
 * @brief Creates a memory region for mmap operations.
 *
 * @param offset The offset within the mmap region.
 * @param size   The size of the region.
 * @return AlignedRegion The aligned memory region.
 */
inline util::AlignedRegion make_mmap_region(size_t offset, size_t size) {
    const auto page_size = static_cast<size_t>(util::get_system_page_size());
    return util::align_region(static_cast<uintptr_t>(offset), size, page_size);
}

/**
 * @brief Creates a memory region for madvise operations.
 *
 * @param data         The base address of the mapped memory region.
 * @param mapping_size The size of the mapped memory region.
 * @param offset       The offset within the mapped memory region.
 * @param size         The size of the region.
 * @return AlignedRegion The aligned memory region.
 */
inline util::AlignedRegion make_madvise_region(const void* data, size_t mapping_size, size_t offset, size_t size) {
    const auto page_size = static_cast<size_t>(util::get_system_page_size());
    if (data == nullptr || mapping_size == 0 || offset >= mapping_size || size < page_size) {
        return {};
    } else {
        const auto available = mapping_size - offset;
        const auto raw_len = (size == auto_size) ? available : std::min(size, available);
        return util::align_region(reinterpret_cast<uintptr_t>(data) + offset, raw_len, page_size);
    }
}
}  // namespace util

class HandleHolder {
    int m_handle = -1;
    void reset() noexcept {
        if (m_handle != -1) {
            close(m_handle);
            m_handle = -1;
        }
    }

public:
    explicit HandleHolder(int handle = -1) : m_handle(handle) {}

    HandleHolder(const HandleHolder&) = delete;
    HandleHolder& operator=(const HandleHolder&) = delete;

    HandleHolder(HandleHolder&& other) noexcept : m_handle(other.m_handle) {
        other.m_handle = -1;
    }

    HandleHolder& operator=(HandleHolder&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        reset();
        m_handle = other.m_handle;
        other.m_handle = -1;
        return *this;
    }

    ~HandleHolder() {
        reset();
    }

    int get() const noexcept {
        return m_handle;
    }
};

class MapHolder final : public MappedMemory {
    void* m_mapped_view = MAP_FAILED;
    size_t m_mapped_view_size = 0;
    size_t m_mapped_view_offset = 0;  // file offset of m_mapped_view
    std::byte* m_data = nullptr;
    size_t m_size = 0;
    std::optional<uint64_t> m_id;
    HandleHolder m_handle;
    // Handles shared with hint_prefetch_async()'s caller; joined before unmapping (see ~MapHolder).
    std::mutex m_pending_prefetch_mutex;
    std::vector<util::PrefetchToken> m_pending_prefetch;

    void adopt_pending_prefetch(util::PrefetchToken task) noexcept {
        if (task.valid()) {
            try {
                std::lock_guard<std::mutex> lock(m_pending_prefetch_mutex);
                // Reap already-finished handles so the vector doesn't grow without bound across repeated
                // hint_prefetch_async() calls over this mapping's lifetime.
                m_pending_prefetch.erase(std::remove_if(m_pending_prefetch.begin(),
                                                        m_pending_prefetch.end(),
                                                        [](util::PrefetchToken& pending) {
                                                            return !pending.valid() ||
                                                                   pending.wait_for(std::chrono::seconds(0)) ==
                                                                       std::future_status::ready;
                                                        }),
                                         m_pending_prefetch.end());
                m_pending_prefetch.push_back(std::move(task));
            } catch (const std::exception&) {
                task.wait();
            }
        }
    }

    void wait_for_pending_prefetch() noexcept {
        std::lock_guard<std::mutex> lock(m_pending_prefetch_mutex);
        for (auto& task : m_pending_prefetch) {
            if (task.valid()) {
                task.wait();
            }
        }
        m_pending_prefetch.clear();
    }

public:
    MapHolder() = default;

    void set(const std::filesystem::path& path, const size_t offset, const size_t size, const MmapMode mmap_mode) {
        int mode = (mmap_mode == MmapMode::READ_WRITE) ? O_RDWR : O_RDONLY;
        int fd = open(path.c_str(), mode);
        if (fd == -1) {
            throw std::runtime_error("Can not open file " + util::path_to_string(path) +
                                     " for mapping. Ensure that file exists and has appropriate permissions.");
        }
        set_from_fd(fd, offset, size, mmap_mode);
        m_id = (mmap_mode == MmapMode::READ_WRITE) ? std::nullopt
                                                   : std::optional<uint64_t>(util::get_id_for_file(path, offset, size));
    }

    void set_from_fd(const int fd, const size_t offset, const size_t size, const MmapMode mmap_mode = MmapMode::READ) {
        m_handle = HandleHolder(fd);

        struct stat sb = {};
        if (fstat(fd, &sb) == -1) {
            throw std::runtime_error("Can not get file size for fd=" + std::to_string(fd));
        }
        const auto file_size = static_cast<size_t>(sb.st_size);
        m_size = (size == auto_size) ? file_size - offset : size;
        if (offset + m_size > file_size || offset + m_size < offset) {
            throw std::runtime_error("Requested mapping range exceeds file size for fd=" + std::to_string(fd));
        }

        if (m_size > 0) {
            const auto prot = (mmap_mode == MmapMode::READ_WRITE) ? (PROT_READ | PROT_WRITE) : PROT_READ;
            const auto& [aligned_offset, length, gap] = util::make_mmap_region(offset, m_size);
            m_mapped_view_size = length;
            m_mapped_view_offset = aligned_offset;
            m_mapped_view = mmap(nullptr, length, prot, MAP_SHARED, fd, aligned_offset);
            if (m_mapped_view == MAP_FAILED) {
                throw std::runtime_error("Can not create file mapping for " + std::to_string(fd) +
                                         ", err=" + std::strerror(errno));
            }
            m_data = static_cast<std::byte*>(m_mapped_view) + gap;
        }
        // A read-write mapping is not an immutable data source, so it must not be shared through id-based caches.
        m_id = (mmap_mode == MmapMode::READ_WRITE)
                   ? std::nullopt
                   : std::optional<uint64_t>(util::u64_hash_combine(static_cast<uint64_t>(sb.st_ino),
                                                                    {static_cast<uint64_t>(sb.st_dev), offset, size}));
    }

    std::optional<uint64_t> get_id() const noexcept override {
        return m_id;
    }

    ~MapHolder() {
        // Detached prefetch tasks may still be touching this mapping's pages; join them first.
        wait_for_pending_prefetch();
        if (m_mapped_view != MAP_FAILED) {
            munmap(m_mapped_view, m_mapped_view_size);
        }
    }

    const std::byte* data() const noexcept override final {
        return m_data;
    }

    std::byte* data() noexcept override final {
        return m_data;
    }

    size_t size() const noexcept override final {
        return m_size;
    }

    void hint_evict(size_t offset, size_t size) noexcept override {
        if (m_mapped_view != MAP_FAILED) {
            if (const auto region = util::make_madvise_region(m_data, m_size, offset, size); region.m_length > 0) {
                std::ignore = madvise(reinterpret_cast<void*>(region.m_address), region.m_length, MADV_DONTNEED);
            }
        }
    }

    void hint_prefetch(size_t offset, size_t size) noexcept override {
        constexpr size_t one_mb = 1024 * 1024;
        // Below 4 MiB the overhead of spawning threads exceeds the benefit; skip.
        if (const auto region = util::make_madvise_region(m_data, m_size, offset, size); region.m_length > 4 * one_mb) {
            const auto num_threads = std::min<size_t>(10, std::thread::hardware_concurrency());
            const auto aligned_size = util::align_size_up(region.m_length, util::get_system_page_size());
            util::vm_prefetch(reinterpret_cast<void*>(region.m_address), aligned_size, num_threads);
        }
    }

    util::PrefetchToken hint_prefetch_async(size_t offset, size_t size) noexcept override {
        if (const auto region = util::clamp_align_region(m_data, m_size, offset, size);
            region.m_length > util::default_parallel_io_threshold) {
            auto future = read_into_page_cache(region);
            adopt_pending_prefetch(future);
            return future;
        } else {
            return {};
        }
    }

private:
    // Reads the region through the file descriptor instead of touching the mapped pages: this fills the page cache
    // without page faults and RSS, and keeps progressing while the process is busy creating other mappings (e.g. a
    // device driver mapping its buffers). The workers take slices front to back and skip already cached ones.
    util::PrefetchToken read_into_page_cache(const util::AlignedRegion& region) noexcept {
        struct Range {
            int fd;
            const std::byte* data;
            size_t file_offset;
            size_t size;
            std::atomic<size_t> next{0};
        };
        static constexpr size_t slice_size = 16 * 1024 * 1024;
        static constexpr size_t read_size = 2 * 1024 * 1024;
        try {
            const auto data = reinterpret_cast<const std::byte*>(region.m_address);
            const auto file_offset =
                m_mapped_view_offset + static_cast<size_t>(data - static_cast<std::byte*>(m_mapped_view));
            const auto end = std::min(file_offset + region.m_length, m_mapped_view_offset + m_mapped_view_size);
            auto range = std::shared_ptr<Range>(new Range{m_handle.get(), data, file_offset, end - file_offset});
            return util::submit_jobs(util::prefetch_thread_count(range->size), [range]() {
                try {
                    const auto page_size = static_cast<size_t>(util::get_system_page_size());
                    std::vector<char> buffer(read_size);
                    std::vector<unsigned char> residency(slice_size / page_size);
                    for (size_t position; (position = range->next.fetch_add(slice_size)) < range->size;) {
                        const auto length = std::min(slice_size, range->size - position);
                        auto address = const_cast<std::byte*>(range->data + position);
                        if (mincore(address, length, residency.data()) == 0 &&
                            std::all_of(residency.begin(),
                                        residency.begin() + util::ceil_div(length, page_size),
                                        [](unsigned char resident) {
                                            return resident & 1;
                                        })) {
                            continue;
                        }
                        for (size_t done = 0; done < length;) {
                            const auto count = pread(range->fd,
                                                     buffer.data(),
                                                     std::min(read_size, length - done),
                                                     static_cast<off_t>(range->file_offset + position + done));
                            if (count <= 0) {
                                break;
                            }
                            done += static_cast<size_t>(count);
                        }
                    }
                } catch (const std::exception&) {
                    // Best effort: the consumer reads the data anyway.
                }
            });
        } catch (const std::exception&) {
            return {};
        }
    }
};

std::shared_ptr<MappedMemory> load_mmap_object(const std::filesystem::path& path,
                                               size_t offset,
                                               size_t size,
                                               bool /* no_placeholder */,
                                               MmapMode mode) {
    auto holder = std::make_shared<MapHolder>();
    holder->set(path, offset, size, mode);
    return holder;
}

std::shared_ptr<ov::MappedMemory> load_mmap_object(FileHandle handle, size_t offset, size_t size) {
    if (handle == -1) {
        throw std::runtime_error("Invalid file descriptor provided for mapping.");
    }
    auto holder = std::make_shared<MapHolder>();
    holder->set_from_fd(handle, offset, size);
    return holder;
}
}  // namespace ov
