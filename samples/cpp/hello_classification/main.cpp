// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
    std::string path = "D:\\opipikin\\models\\TinyLlama-1.1B-Chat-v1.0\\openvino_model.xml";
    //ov::Core core;
    //ov::CompiledModel compiled_model;
    //core.set_property(ov::cache_dir("D:\\opipikin\\model_cache"));

    // core.set_property(ov::enable_mmap(false));
    std::chrono::steady_clock::time_point begin, end;

    for (int i = 0; i < 20; i++) {
        ov::Core core;
        core.set_property(ov::cache_dir("D:\\opipikin\\model_cache"));

        //std::cout << "try: " << i << "\n";
        begin = std::chrono::steady_clock::now();
        //auto model = core.read_model(path);
        // compiled_model = core.compile_model(model, "CPU");
        auto compiled_model = core.compile_model(path, "GPU");
        end = std::chrono::steady_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "\n";

    }
    return EXIT_SUCCESS;
}
