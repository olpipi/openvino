// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/split_concat_memory.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(SplitConcatMemory, cyclicBufferCorrectness) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    ov::Core core;

    auto compiled_model = core.compile_model(function, targetDevice);
    auto infer_request = compiled_model.create_infer_request();

    /*
     * cnc1 out  |  mem      | In|q
     *           |===============|
     * iter_1    | 0 | 0 | 0 | 1 |
     * iter_2    | 0 | 0 | 1 | 2 |
     * iter 3    | 0 | 1 | 2 | 3 |
     */

    auto input_tensor = infer_request.get_tensor("input_t");
    auto output_tensor = infer_request.get_tensor("plus_one_t");

    auto output_tensor_ref = ov::Tensor(output_tensor.get_element_type(), output_tensor.get_shape());

    auto fill_by_quarter = [this] (ov::Tensor& tensor, std::vector<float> vals) {
        OPENVINO_ASSERT(vals.size() == 4);
        auto quarter_blocked_shape = tensor.get_shape();

        // splis axis dimension into chunk
        OPENVINO_ASSERT(quarter_blocked_shape[axis] % vals.size() == 0);
        quarter_blocked_shape[axis] /= vals.size();
        quarter_blocked_shape.insert(quarter_blocked_shape.begin() + axis, vals.size());

        auto quarter_blocked_view = CommonTestUtils::make_reshape_view(tensor, quarter_blocked_shape);
        CommonTestUtils::fill_data_with_broadcast(quarter_blocked_view, axis, vals);
    };

    // iteration 1

    CommonTestUtils::fill_data_const(input_tensor, 1);
    fill_by_quarter(output_tensor_ref, {1, 1, 1, 2});
    infer_request.infer();
    Compare(output_tensor_ref, output_tensor);

    // iteration 2
    CommonTestUtils::fill_data_const(input_tensor, 2);
    fill_by_quarter(output_tensor_ref, {1, 1, 2, 3});
    infer_request.infer();
    Compare(output_tensor_ref, output_tensor);

    // iteration 3
    CommonTestUtils::fill_data_const(input_tensor, 3);
    fill_by_quarter(output_tensor_ref, {1, 2, 3, 4});
    infer_request.infer();
    Compare(output_tensor_ref, output_tensor);
}

}  // namespace SubgraphTestsDefinitions
