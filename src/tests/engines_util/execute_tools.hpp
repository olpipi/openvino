// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <list>
#include <memory>
#include <random>
#include <vector>

#include "gtest/gtest.h"
#include "ngraph/file_util.hpp"
#include "ngraph/function.hpp"
#include "ngraph/log.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "ngraph/type/element_type_traits.hpp"

namespace ngraph {
class TestOpMultiOut : public op::Op {
public:
    OPENVINO_OP("TestOpMultiOut");
    TestOpMultiOut() = default;

    TestOpMultiOut(const Output<Node>& output_1, const Output<Node>& output_2);
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    OPENVINO_SUPPRESS_DEPRECATED_START
    bool evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const override;
    OPENVINO_SUPPRESS_DEPRECATED_END
};
}  // namespace ngraph

bool validate_list(const std::vector<std::shared_ptr<ngraph::Node>>& nodes);
std::shared_ptr<ngraph::Function> make_test_graph();

template <typename T>
void copy_data(std::shared_ptr<ngraph::runtime::Tensor> tv, const std::vector<T>& data) {
    size_t data_size = data.size() * sizeof(T);
    if (data_size > 0) {
        tv->write(data.data(), data_size);
    }
}

template <>
void copy_data<bool>(std::shared_ptr<ngraph::runtime::Tensor> tv, const std::vector<bool>& data) {
    std::vector<char> data_char(data.begin(), data.end());
    copy_data(tv, data_char);
}

template <ngraph::element::Type_t ET>
ngraph::HostTensorPtr make_host_tensor(const ngraph::Shape& shape,
                                       const std::vector<typename ngraph::element_type_traits<ET>::value_type>& data) {
    NGRAPH_CHECK(shape_size(shape) == data.size(), "Incorrect number of initialization elements");
    auto host_tensor = std::make_shared<ngraph::HostTensor>(ET, shape);
    copy_data(host_tensor, data);
    return host_tensor;
}

void random_init(ngraph::runtime::Tensor* tv, std::default_random_engine& engine);

template <ngraph::element::Type_t ET>
ngraph::HostTensorPtr make_host_tensor(const ngraph::Shape& shape) {
    auto host_tensor = std::make_shared<ngraph::HostTensor>(ET, shape);
    static std::default_random_engine engine(2112);
    random_init(host_tensor.get(), engine);
    return host_tensor;
}

testing::AssertionResult test_ordered_ops(std::shared_ptr<ngraph::Function> f, const ngraph::NodeVector& required_ops);
