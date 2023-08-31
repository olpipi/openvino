// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <tuple>
#include <string>
#include <map>
#include <memory>
#include <set>
#include <functional>
#include <gtest/gtest.h>

#include "ie_core.hpp"
#include "ie_precision.hpp"

#include "functional_test_utils/blob_utils.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "common_test_utils/common_utils.hpp"

#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"

#include "common_test_utils/test_enums.hpp"

namespace LayerTestsDefinitions {

static std::map<ngraph::helpers::ActivationTypes, std::string> activationNames = {
        {ngraph::helpers::ActivationTypes::Sigmoid,               "Sigmoid"},
        {ngraph::helpers::ActivationTypes::Tanh,                  "Tanh"},
        {ngraph::helpers::ActivationTypes::Relu,                  "Relu"},
        {ngraph::helpers::ActivationTypes::LeakyRelu,             "LeakyRelu"},
        {ngraph::helpers::ActivationTypes::Exp,                   "Exp"},
        {ngraph::helpers::ActivationTypes::Log,                   "Log"},
        {ngraph::helpers::ActivationTypes::Sign,                  "Sign"},
        {ngraph::helpers::ActivationTypes::Abs,                   "Abs"},
        {ngraph::helpers::ActivationTypes::Clamp,                 "Clamp"},
        {ngraph::helpers::ActivationTypes::Negative,              "Negative"},
        {ngraph::helpers::ActivationTypes::Acos,                  "Acos"},
        {ngraph::helpers::ActivationTypes::Acosh,                 "Acosh"},
        {ngraph::helpers::ActivationTypes::Asin,                  "Asin"},
        {ngraph::helpers::ActivationTypes::Asinh,                 "Asinh"},
        {ngraph::helpers::ActivationTypes::Atan,                  "Atan"},
        {ngraph::helpers::ActivationTypes::Atanh,                  "Atanh"},
        {ngraph::helpers::ActivationTypes::Cos,                   "Cos"},
        {ngraph::helpers::ActivationTypes::Cosh,                  "Cosh"},
        {ngraph::helpers::ActivationTypes::Floor,                 "Floor"},
        {ngraph::helpers::ActivationTypes::Sin,                   "Sin"},
        {ngraph::helpers::ActivationTypes::Sinh,                  "Sinh"},
        {ngraph::helpers::ActivationTypes::Sqrt,                  "Sqrt"},
        {ngraph::helpers::ActivationTypes::Tan,                   "Tan"},
        {ngraph::helpers::ActivationTypes::Elu,                   "Elu"},
        {ngraph::helpers::ActivationTypes::Erf,                   "Erf"},
        {ngraph::helpers::ActivationTypes::HardSigmoid,           "HardSigmoid"},
        {ngraph::helpers::ActivationTypes::Selu,                  "Selu"},
        {ngraph::helpers::ActivationTypes::Sigmoid,               "Sigmoid"},
        {ngraph::helpers::ActivationTypes::Tanh,                  "Tanh"},
        {ngraph::helpers::ActivationTypes::Relu,                  "Relu"},
        {ngraph::helpers::ActivationTypes::LeakyRelu,             "LeakyRelu"},
        {ngraph::helpers::ActivationTypes::Exp,                   "Exp"},
        {ngraph::helpers::ActivationTypes::Log,                   "Log"},
        {ngraph::helpers::ActivationTypes::Sign,                  "Sign"},
        {ngraph::helpers::ActivationTypes::Abs,                   "Abs"},
        {ngraph::helpers::ActivationTypes::Gelu,                  "Gelu"},
        {ngraph::helpers::ActivationTypes::Ceiling,               "Ceiling"},
        {ngraph::helpers::ActivationTypes::PReLu,                 "PReLu"},
        {ngraph::helpers::ActivationTypes::Mish,                  "Mish"},
        {ngraph::helpers::ActivationTypes::HSwish,                "HSwish"},
        {ngraph::helpers::ActivationTypes::SoftPlus,              "SoftPlus"},
        {ngraph::helpers::ActivationTypes::Swish,                 "Swish"},
        {ngraph::helpers::ActivationTypes::HSigmoid,              "HSigmoid"},
        {ngraph::helpers::ActivationTypes::RoundHalfToEven,       "RoundHalfToEven"},
        {ngraph::helpers::ActivationTypes::RoundHalfAwayFromZero, "RoundHalfAwayFromZero"},
        {ngraph::helpers::ActivationTypes::GeluErf,               "GeluErf"},
        {ngraph::helpers::ActivationTypes::GeluTanh,              "GeluTanh"},
        {ngraph::helpers::ActivationTypes::SoftSign,              "SoftSign"},
};

typedef std::tuple<
        std::pair<ngraph::helpers::ActivationTypes, std::vector<float>>, // Activation type and constant value
        InferenceEngine::Precision,
        InferenceEngine::Precision,    // Input precision
        InferenceEngine::Precision,    // Output precision
        InferenceEngine::Layout,       // Input layout
        InferenceEngine::Layout,       // Output layout
        std::pair<std::vector<size_t>, std::vector<size_t>>,
        std::string> activationParams;

class ActivationLayerTest : public testing::WithParamInterface<activationParams>,
                            virtual public LayerTestsUtils::LayerTestsCommon {
public:
    ngraph::helpers::ActivationTypes activationType;
    static std::string getTestCaseName(const testing::TestParamInfo<activationParams> &obj);
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo &info) const override;

protected:
    void SetUp() override;
};

class ActivationParamLayerTest : public ActivationLayerTest {
protected:
    void SetUp() override;

private:
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo &info) const override;
    void generateActivationBlob(std::vector<float> constantsValue);
    ngraph::ParameterVector createActivationParams(
        ngraph::element::Type ngPrc, std::vector<size_t> inShape = {});

private:
    std::vector<float> constantsValue;
};

class ActivationDynamicLayerTest : public ActivationLayerTest {
public:
    std::unordered_set<size_t> static_dims;
    void Run() override;
};

}  // namespace LayerTestsDefinitions

namespace ov {
namespace test {
using ov::test::utils::ActivationTypes;

static std::map<ActivationTypes, std::string> activationNames = {
        {ActivationTypes::Sigmoid,               "Sigmoid"},
        {ActivationTypes::Tanh,                  "Tanh"},
        {ActivationTypes::Relu,                  "Relu"},
        {ActivationTypes::LeakyRelu,             "LeakyRelu"},
        {ActivationTypes::Exp,                   "Exp"},
        {ActivationTypes::Log,                   "Log"},
        {ActivationTypes::Sign,                  "Sign"},
        {ActivationTypes::Abs,                   "Abs"},
        {ActivationTypes::Clamp,                 "Clamp"},
        {ActivationTypes::Negative,              "Negative"},
        {ActivationTypes::Acos,                  "Acos"},
        {ActivationTypes::Acosh,                 "Acosh"},
        {ActivationTypes::Asin,                  "Asin"},
        {ActivationTypes::Asinh,                 "Asinh"},
        {ActivationTypes::Atan,                  "Atan"},
        {ActivationTypes::Atanh,                  "Atanh"},
        {ActivationTypes::Cos,                   "Cos"},
        {ActivationTypes::Cosh,                  "Cosh"},
        {ActivationTypes::Floor,                 "Floor"},
        {ActivationTypes::Sin,                   "Sin"},
        {ActivationTypes::Sinh,                  "Sinh"},
        {ActivationTypes::Sqrt,                  "Sqrt"},
        {ActivationTypes::Tan,                   "Tan"},
        {ActivationTypes::Elu,                   "Elu"},
        {ActivationTypes::Erf,                   "Erf"},
        {ActivationTypes::HardSigmoid,           "HardSigmoid"},
        {ActivationTypes::Selu,                  "Selu"},
        {ActivationTypes::Sigmoid,               "Sigmoid"},
        {ActivationTypes::Tanh,                  "Tanh"},
        {ActivationTypes::Relu,                  "Relu"},
        {ActivationTypes::LeakyRelu,             "LeakyRelu"},
        {ActivationTypes::Exp,                   "Exp"},
        {ActivationTypes::Log,                   "Log"},
        {ActivationTypes::Sign,                  "Sign"},
        {ActivationTypes::Abs,                   "Abs"},
        {ActivationTypes::Gelu,                  "Gelu"},
        {ActivationTypes::Ceiling,               "Ceiling"},
        {ActivationTypes::PReLu,                 "PReLu"},
        {ActivationTypes::Mish,                  "Mish"},
        {ActivationTypes::HSwish,                "HSwish"},
        {ActivationTypes::SoftPlus,              "SoftPlus"},
        {ActivationTypes::Swish,                 "Swish"},
        {ActivationTypes::HSigmoid,              "HSigmoid"},
        {ActivationTypes::RoundHalfToEven,       "RoundHalfToEven"},
        {ActivationTypes::RoundHalfAwayFromZero, "RoundHalfAwayFromZero"},
        {ActivationTypes::GeluErf,               "GeluErf"},
        {ActivationTypes::GeluTanh,              "GeluTanh"},
        {ActivationTypes::SoftSign,              "SoftSign"},
};

typedef std::tuple<
        std::pair<ActivationTypes, std::vector<float>>, // Activation type and constant value
        ov::element::Type,
        ov::element::Type,    // Input precision
        ov::element::Type,    // Output precision
        std::pair<std::vector<size_t>, std::vector<size_t>>,
        std::string> activationParams;

class ActivationLayerTestNew : public testing::WithParamInterface<activationParams>,
                            virtual public ov::test::SubgraphBaseTest {
public:
    ActivationTypes activationType;
    static std::string getTestCaseName(const testing::TestParamInfo<activationParams> &obj);
protected:
    void SetUp() override;
};

class ActivationParamLayerTestNew : public ActivationLayerTestNew {
protected:
    void SetUp() override;

private:
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
    void generateActivationBlob(std::vector<float> constantsValue);
    ov::ParameterVector createActivationParams(ov::element::Type type, std::vector<size_t> inShape = {});

private:
    std::vector<float> constantsValue;
};

class ActivationDynamicLayerTestNew : public ActivationLayerTestNew {
public:
    std::unordered_set<size_t> static_dims;
    void run() override;
};

} //  namespace test
} //  namespace ov