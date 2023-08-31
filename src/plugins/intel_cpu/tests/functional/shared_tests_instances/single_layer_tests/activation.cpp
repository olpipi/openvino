// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "single_layer_tests/activation.hpp"
#include "common_test_utils/test_constants.hpp"

using ov::test::utils::ActivationTypes;
using ov::test::activationNames;
using ov::test::ActivationLayerTestNew;
using ov::test::ActivationParamLayerTestNew;
using ov::test::ActivationDynamicLayerTestNew;

namespace {
// Common params
const std::vector<ov::element::Type> inputPrecisions = {
        ov::element::f32
        // TODO: Fix Issue-27390
        // InferenceEngine::Precision::I16,
        // InferenceEngine::Precision::U8
};

const std::vector<ov::element::Type> netPrecisions = {
        ov::element::f32,
        ov::element::f16
};

const std::vector<ov::element::Type> intPrecisions = {
        ov::element::i32,
};

const std::map<ActivationTypes, std::vector<std::vector<float>>> activationTypes = {
        {ActivationTypes::Sigmoid,               {}},
        {ActivationTypes::Tan,                   {}},
        {ActivationTypes::Tanh,                  {}},
        {ActivationTypes::Relu,                  {}},
        {ActivationTypes::Exp,                   {}},
        {ActivationTypes::Log,                   {}},
        {ActivationTypes::Sign,                  {}},
        {ActivationTypes::Abs,                   {}},
        {ActivationTypes::Clamp,                 {{-2.0f, 2.0f}}},
        {ActivationTypes::Negative,              {}},
        {ActivationTypes::Acos,                  {}},
        {ActivationTypes::Acosh,                  {}},
        {ActivationTypes::Asin,                  {}},
        {ActivationTypes::Asinh,                 {}},
        {ActivationTypes::Atan,                  {}},
        {ActivationTypes::Atanh,                  {}},
        {ActivationTypes::Cos,                   {}},
        {ActivationTypes::Cosh,                  {}},
        {ActivationTypes::Floor,                 {}},
        {ActivationTypes::Sin,                   {}},
        {ActivationTypes::Sinh,                  {}},
        {ActivationTypes::Sqrt,                  {}},
        {ActivationTypes::Elu,                   {{0.1f}}},
        {ActivationTypes::Erf,                   {}},
        {ActivationTypes::HardSigmoid,           {{0.2f, 0.5f}}},
        {ActivationTypes::Selu,                  {{1.6732f, 1.0507f}}},
        {ActivationTypes::Ceiling,               {}},
        {ActivationTypes::Mish,                  {}},
        {ActivationTypes::HSwish,                {}},
        {ActivationTypes::SoftPlus,              {}},
        {ActivationTypes::HSigmoid,              {}},
        {ActivationTypes::RoundHalfToEven,       {}},
        {ActivationTypes::RoundHalfAwayFromZero, {}},
        {ActivationTypes::GeluErf,               {}},
        {ActivationTypes::GeluTanh,              {}},
        {ActivationTypes::Swish,                 {{0.4f}}}
};

// List of operations that should be tested also with integer precision
const std::map<ActivationTypes, std::vector<std::vector<float>>> intActivationTypes = {
        {ActivationTypes::Acosh,                 {}},
        {ActivationTypes::Asinh,                 {}},
        {ActivationTypes::Atan,                  {}},
        {ActivationTypes::Negative,              {}},
        {ActivationTypes::Ceiling,               {}},
        {ActivationTypes::Cos,                   {}},
        {ActivationTypes::Cosh,                  {}},
        {ActivationTypes::Sign,                  {}},
        {ActivationTypes::Sinh,                  {}},
        {ActivationTypes::Sqrt,                  {}},
        {ActivationTypes::Tan,                   {}},
        {ActivationTypes::Tanh,                  {}},
};

const std::map<ActivationTypes, std::vector<std::vector<float>>> activationParamTypes = {
        {ActivationTypes::PReLu, {{}}}, // Slope will be filled with increasing values from -10 to match slope input shape
        {ActivationTypes::LeakyRelu, {{0.01f}}}
};

std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> basic = {
        {{1, 50}, {{}}},
        {{5, 128}, {{}}},
        {{2, 2, 2, 2, 2, 2, 2, 2}, {{}}},
};

std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> preluBasic = {
        {{1, 50}, {{1}, {50}}},
        {{1, 128}, {{1}, {128}}},

        // Broadcast check
        {{3, 2}, {{1}, {2}, {3, 2}}},
        {{3, 2, 5}, {{1}, {2}, {5}, {2, 5}, {3, 1, 5}, {1, 2, 1}, {1, 1, 5}, {3, 1, 1}, {3, 2, 5}}},
        {{2, 1, 2}, {{2}, {2, 1, 1}}},
        {{3, 2, 5, 7}, {{1}, {7}, {2}, {5, 7}, {2, 5, 7}, {2, 1, 1}, {1, 2, 1, 1}, {3, 2, 1, 1}, {3, 2, 5, 7}}},
        {{2, 2, 2, 2, 2, 2, 2, 2}, {{2}, {2, 2}, {2, 1, 1, 2}}},
};

const auto basicCases = ::testing::Combine(
        ::testing::ValuesIn(ov::test::utils::combineParams(activationTypes)),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(ov::element::undefined),
        ::testing::Values(ov::element::undefined),
        ::testing::ValuesIn(ov::test::utils::combineParams(basic)),
        ::testing::Values(ov::test::utils::DEVICE_CPU)
);

const auto basicPreluCases = ::testing::Combine(
        ::testing::ValuesIn(ov::test::utils::combineParams(activationParamTypes)),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(ov::element::undefined),
        ::testing::Values(ov::element::undefined),
        ::testing::ValuesIn(ov::test::utils::combineParams(preluBasic)),
        ::testing::Values(ov::test::utils::DEVICE_CPU)
);

const auto basicIntegerOperations = ::testing::Combine(
            ::testing::ValuesIn(ov::test::utils::combineParams(intActivationTypes)),
            ::testing::ValuesIn(intPrecisions),
            ::testing::ValuesIn(intPrecisions),
            ::testing::ValuesIn(intPrecisions),
            ::testing::ValuesIn(ov::test::utils::combineParams(basic)),
            ::testing::Values(ov::test::utils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_Activation_Basic, ActivationLayerTestNew, basicCases, ActivationLayerTestNew::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Activation_Basic, ActivationDynamicLayerTestNew, basicCases, ActivationLayerTestNew::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Integer_Activation_Basic, ActivationLayerTestNew, basicIntegerOperations, ActivationLayerTestNew::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Activation_Basic_Prelu_Const, ActivationLayerTestNew, basicPreluCases, ActivationLayerTestNew::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Activation_Basic_Prelu_Param, ActivationParamLayerTestNew, basicPreluCases, ActivationLayerTestNew::getTestCaseName);
}  // namespace
