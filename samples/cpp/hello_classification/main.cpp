// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <chrono>
#include <cstring>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

// clang-format off
#include "openvino/openvino.hpp"

#include "samples/args_helper.hpp"
#include "samples/common.hpp"
#include "samples/classification_results.h"
#include "samples/slog.hpp"
#include "format_reader_ptr.h"
// clang-format on

/**
 * @brief Main with support Unicode paths, wide strings
 */
int tmain(int argc, tchar* argv[]) {
    const std::string model_path =
        argc > 1 ? argv[1] : "/opt/home/opipikin/workspace/models/TinyLlama-1.1B-Chat-v1.0/openvino_model.xml";
    const std::string device = argc > 2 ? argv[2] : "GPU.0";
    const std::string cache_dir = argc > 3 ? argv[3] : "/opt/home/opipikin/workspace/models/cache";

    using clock = std::chrono::high_resolution_clock;
    auto ms_since = [](clock::time_point start) {
        return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(clock::now() - start).count();
    };

    ov::Core core;
    std::cout << "device: " << core.get_property(device, ov::device::full_name) << std::endl;

    std::cout << "\n";
    auto t0 = clock::now();
    ov::CompiledModel compiled_model = core.compile_model(model_path, device, ov::AnyMap{ov::cache_dir(cache_dir)});
    std::cout << "compile_model:       " << ms_since(t0) << " ms" << std::endl;
    std::cout << "loaded_from_cache: " << compiled_model.get_property(ov::loaded_from_cache) << std::endl;

    t0 = clock::now();
    ov::InferRequest infer_request = compiled_model.create_infer_request();
    std::cout << "create_infer_request:" << ms_since(t0) << " ms" << std::endl;

    // fill every input with zeros (not just allocate), replacing dynamic dimensions with 1;
    // required because e.g. beam_idx must be a valid index, not left as uninitialized memory
    auto fill_inputs_with_zeros = [&]() {
        for (const auto& input : compiled_model.inputs()) {
            ov::Shape shape;
            for (const auto& dim : input.get_partial_shape())
                shape.push_back(dim.is_static() ? dim.get_length() : 1);
            ov::Tensor tensor(input.get_element_type(), shape);
            std::memset(tensor.data(), 0, tensor.get_byte_size());
            infer_request.set_tensor(input, tensor);
        }
    };

    t0 = clock::now();
    fill_inputs_with_zeros();
    std::cout << "fill_input_data:     " << ms_since(t0) << " ms" << std::endl;

    t0 = clock::now();
    infer_request.infer();
    std::cout << "infer:               " << ms_since(t0) << " ms" << std::endl;

    t0 = clock::now();
    fill_inputs_with_zeros();

    t0 = clock::now();
    infer_request.infer();
    std::cout << "infer 2:               " << ms_since(t0) << " ms" << std::endl;

    t0 = clock::now();
    fill_inputs_with_zeros();

    t0 = clock::now();
    infer_request.infer();
    std::cout << "infer 3:               " << ms_since(t0) << " ms" << std::endl;

    return EXIT_SUCCESS;
}
