// Copyright (C) 2018-2025 Intel Corporation
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
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/op_conversions/convert_interpolate11_downgrade.hpp"
// clang-format on

/**
 * @brief Main with support Unicode paths, wide strings
 */
int tmain(int argc, tchar* argv[]) {
    ov::Core core;
    auto model = core.read_model("../../../new_out.xml");

    ov::pass::Manager manager;
    manager.register_pass<ov::pass::ConvertInterpolate11ToInterpolate4>();
    manager.run_passes(model);
    ov::pass::ConstantFolding().run_on_model(model);

    return EXIT_SUCCESS;
}
