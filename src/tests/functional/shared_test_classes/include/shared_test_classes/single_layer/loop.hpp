// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include <ngraph/op/util/attr_types.hpp>
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

namespace LayerTestsDefinitions {
enum LOOP_IN_TYPE {
    INVARIANT,
    MERGED
};

using LoopParams = typename std::tuple<
        bool,                                                              // ExecuteFirstIteration
        bool,                                                              // BodyCondition is a constant?
        bool,                                                              // BodyCondition value, if it is a Const
        int64_t,                                                           // TripCount, -1 means infinity
        std::vector<std::pair<std::vector<size_t>, LOOP_IN_TYPE>>,         // inputs
        ov::element::Type_t,                                        // Network precision
        std::string>;                                                      // Device name

class LoopTest : public testing::WithParamInterface<LoopParams>,
                 virtual public LayerTestsUtilsNew::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<LoopParams> &obj);

protected:
    void SetUp() override;
};


using StaticShapeLoopParams = typename std::tuple<
        bool,
        bool,
        std::tuple<
            bool,
            int64_t,
            int64_t,
            int64_t
            >,
        int64_t,
        CommonTestUtils::SizeVector,
        ov::element::Type_t,
        std::string,
        ov::AnyMap
        >;

/**
 * Test case with static SHAPE version of loop operation.
 * Total iteration count is dynamic.
 */
class StaticShapeLoopTest : public testing::WithParamInterface<StaticShapeLoopParams>,
                            virtual public LayerTestsUtilsNew::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<StaticShapeLoopParams> &obj);
    ov::Tensor GenerateInput(ov::element::Type prc, const ov::Shape& shape) const override;
    std::vector<std::pair<ov::element::Type, std::vector<std::uint8_t>>> PredefinedRefs();

private:
    bool unrolling;             // unroll Loop
    bool static_iter_num;       // trip count provided by constant node
    bool static_continue_cond;  // initial_cond provided by constant node
    int64_t max_iter_num;       // -1 means infinity loop (expected dynamic exit condition in body)
    int64_t dynamic_exit;       // -1 means always true
    int64_t axis;               // -1 means no auto concatenation
    int64_t start_value;
    CommonTestUtils::SizeVector data_shape;
    ngraph::element::Type_t data_prc;

    int64_t actual_n_iter();

protected:
    void SetUp() override;
};


class TrivialLoopTest : public testing::WithParamInterface<LayerTestsUtilsNew::basicParamsNew>,
                        virtual public LayerTestsUtilsNew::LayerTestsCommon {
protected:
    using RefTensorGenerator = std::function<ov::Tensor (ov::element::Type prc, const ov::Shape& shape)>;
    RefTensorGenerator inputGen, outputGen;

    void CreateSlicedLoop(size_t batch_size, size_t num_iteration, ov::element::Type prc,
                          CommonTestUtils::SizeVector& shape_vector);
    void CreateSlicedLoopDynCondition(size_t batch_size, size_t num_iteration, ov::element::Type prc,
                          CommonTestUtils::SizeVector& shape_vector, size_t trip_count);

    ov::Tensor GenerateInput(ov::element::Type prc, const ov::Shape& shape) const override {
        if (!!inputGen) {
            return inputGen(prc, shape);
        }

        return LayerTestsCommon::GenerateInput(prc, shape);
    }

    std::vector<std::pair<ov::element::Type, std::vector<std::uint8_t>>> CalculateRefs() override {
        if (!outputGen)
            return LayerTestsCommon::CalculateRefs();

        const auto& outputs = GetOutputs();

        std::vector<std::pair<ov::element::Type, std::vector<std::uint8_t>>> res_collection(outputs.size());

        for (size_t i = 0; i < outputs.size(); i++) {
            const auto& output = outputs[i];
            outputGen(output.get_element_type(), output.get_shape());
            auto data_ptr = static_cast<uint8_t *>(output.data());
            auto data_size = output.get_byte_size();

            auto &res = res_collection[i];
            res.first = output.get_element_type();
            res.second.resize(data_size);
            std::copy(data_ptr, data_ptr + data_size, res.second.begin());
        }
        return res_collection;
    }
};

}  // namespace LayerTestsDefinitions
