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
#include <chrono>

/**
 * @brief Main with support Unicode paths, wide strings
 */
int tmain(int argc, tchar* argv[]) {

    ov::Core core;
    core.set_property(ov::cache_dir("/home/oleg/workspace/models/cache"));
    core.set_property(ov::enable_mmap(true));

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    core.compile_model("/home/oleg/workspace/models/Qwen-7B-Chat/openvino_model.xml", "GPU");
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "\n";

    return EXIT_SUCCESS;
}
