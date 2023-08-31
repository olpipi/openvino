// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ostream>

namespace ov {
namespace test {
namespace utils {

enum ActivationTypes {
    None,
    Sigmoid,
    Tanh,
    Relu,
    LeakyRelu,
    Exp,
    Log,
    Sign,
    Abs,
    Gelu,
    Clamp,
    Negative,
    Acos,
    Acosh,
    Asin,
    Asinh,
    Atan,
    Atanh,
    Cos,
    Cosh,
    Floor,
    Sin,
    Sinh,
    Sqrt,
    Tan,
    Elu,
    Erf,
    HardSigmoid,
    Selu,
    Ceiling,
    PReLu,
    Mish,
    HSwish,
    SoftPlus,
    Swish,
    HSigmoid,
    RoundHalfToEven,
    RoundHalfAwayFromZero,
    GeluErf,
    GeluTanh,
    SoftSign
};

enum class ComparisonTypes {
    EQUAL,
    NOT_EQUAL,
    IS_FINITE,
    IS_INF,
    IS_NAN,
    LESS,
    LESS_EQUAL,
    GREATER,
    GREATER_EQUAL
};

enum class ConversionTypes {
    CONVERT,
    CONVERT_LIKE
};

enum class ReductionType {
    Mean,
    Max,
    Min,
    Prod,
    Sum,
    LogicalOr,
    LogicalAnd,
    L1,
    L2
};

enum class InputLayerType {
    CONSTANT,
    PARAMETER,
};

enum class SequenceTestsMode {
    PURE_SEQ,
    PURE_SEQ_RAND_SEQ_LEN_CONST,
    PURE_SEQ_RAND_SEQ_LEN_PARAM,
    CONVERT_TO_TI_MAX_SEQ_LEN_CONST,
    CONVERT_TO_TI_MAX_SEQ_LEN_PARAM,
    CONVERT_TO_TI_RAND_SEQ_LEN_CONST,
    CONVERT_TO_TI_RAND_SEQ_LEN_PARAM,
};

std::ostream& operator<<(std::ostream& os, const ComparisonTypes type);

std::ostream& operator<<(std::ostream& os, const ConversionTypes type);

std::ostream &operator<<(std::ostream& os, const ReductionType type);

std::ostream& operator<<(std::ostream& os, const InputLayerType type);

std::ostream& operator<<(std::ostream& os, const SequenceTestsMode type);

}  // namespace utils
}  // namespace test
}  // namespace ov
