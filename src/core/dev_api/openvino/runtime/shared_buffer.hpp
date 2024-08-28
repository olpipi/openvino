// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/runtime/aligned_buffer.hpp"
#include "openvino/util/mmap_object.hpp"

namespace ov {

/// \brief SharedBuffer class to store pointer to pre-acclocated buffer.
template <typename T>
class SharedBuffer : public ov::AlignedBuffer {
public:
    SharedBuffer(char* data, size_t size, const T& shared_object) : _shared_object(shared_object) {
        m_allocated_buffer = data;
        m_aligned_buffer = data;
        m_byte_size = size;
    }

    virtual ~SharedBuffer() {
        m_aligned_buffer = nullptr;
        m_allocated_buffer = nullptr;
        m_byte_size = 0;
    }

private:
    T _shared_object;
};

class SharedAsyncBuffer : public ov::AlignedBuffer {
public:
    size_t size() const override {
        update_buffer_if_required();
        return m_byte_size;
    }
    void* get_ptr(size_t offset) const override{
        update_buffer_if_required();
        return m_aligned_buffer + offset;
    }
    void* get_ptr() override{
        update_buffer_if_required();
        return m_aligned_buffer;
    }
    const void* get_ptr() const override{
        update_buffer_if_required();
        return m_aligned_buffer;
    }
    template <typename T>
    T* get_ptr() {
        update_buffer_if_required();
        return reinterpret_cast<T*>(m_aligned_buffer);
    }
    template <typename T>
    const T* get_ptr() const {
        update_buffer_if_required();
        return reinterpret_cast<const T*>(m_aligned_buffer);
    }

    SharedAsyncBuffer(std::shared_ptr<AsyncMemHolder> shared_object) : m_shared_object(shared_object) {};

    virtual ~SharedAsyncBuffer() {
        m_aligned_buffer = nullptr;
        m_allocated_buffer = nullptr;
        m_byte_size = 0;
    }

private:
    void update_buffer_if_required() const {
        std::call_once(mem_is_ready, [this] {
            m_allocated_buffer = m_shared_object->data();
            m_aligned_buffer = m_allocated_buffer;
            m_byte_size = m_shared_object->size();
        });
    };

    mutable std::once_flag mem_is_ready;
    std::shared_ptr<AsyncMemHolder> m_shared_object;
};

}  // namespace ov
