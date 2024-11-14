// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <type_traits>
#include "intel_gpu/graph/serialization/binary_buffer.hpp"
#include <chrono>

namespace cldnn {

    void BinaryInputBuffer::read(void* const data, std::streamsize size) {
        std::chrono::steady_clock::time_point begin, end;

        begin = std::chrono::steady_clock::now();
        auto const read_size = _stream.rdbuf()->sgetn(reinterpret_cast<char*>(data), size);
        end = std::chrono::steady_clock::now();
        OPENVINO_ASSERT(read_size == size,
            "[GPU] Failed to read " + std::to_string(size) + " bytes from stream! Read " + std::to_string(read_size));

        static int i = 0;
        static std::chrono::steady_clock::duration result;
        if (i++) {
            result += end - begin;
        } else {
            result = end - begin;
        }

        //std::cout << "sum read: " << std::chrono::duration_cast<std::chrono::milliseconds>(result).count() << "\n";
    }
}  // namespace cldnn
