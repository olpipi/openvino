// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/experimental_detectron_detection_output.hpp"

namespace ov {
namespace test {
using Attributes = ov::op::v6::ExperimentalDetectronDetectionOutput::Attributes;

namespace {
    std::ostream& operator <<(std::ostream& ss, const Attributes& attributes) {
    ss << "score_threshold=" << attributes.score_threshold << "_";
    ss << "nms_threshold=" << attributes.nms_threshold << "_";
    ss << "max_delta_log_wh=" << attributes.max_delta_log_wh << "_";
    ss << "num_classes=" << attributes.num_classes << "_";
    ss << "post_nms_count=" << attributes.post_nms_count << "_";
    ss << "max_detections_per_image=" << attributes.max_detections_per_image << "_";
    ss << "class_agnostic_box_regression=" << (attributes.class_agnostic_box_regression ? "true" : "false") << "_";
    ss << "deltas_weights=" << ov::test::utils::vec2str(attributes.deltas_weights);
    return ss;
}
} // namespace

std::string ExperimentalDetectronDetectionOutputLayerTest::getTestCaseName(
        const testing::TestParamInfo<ExperimentalDetectronDetectionOutputTestParams>& obj) {
    std::vector<ov::test::InputShape> shapes;
    Attributes attributes;
    ElementType model_type;
    std::string targetName;
    std::tie(
        shapes,
        attributes.score_threshold,
        attributes.nms_threshold,
        attributes.max_delta_log_wh,
        attributes.num_classes,
        attributes.post_nms_count,
        attributes.max_detections_per_image,
        attributes.class_agnostic_box_regression,
        attributes.deltas_weights,
        model_type,
        targetName) = obj.param;

    std::ostringstream result;
    result << "input_rois=" << shapes[0] << "_";
    result << "input_deltas=" << shapes[1] << "_";
    result << "input_scores=" << shapes[2] << "_";
    result << "input_im_info=" << shapes[3] << "_";

    result << "attributes={" << attributes << "}_";
    result << "netPRC=" << model_type << "_";
    result << "trgDev=" << targetName;
    return result.str();
}

void ExperimentalDetectronDetectionOutputLayerTest::SetUp() {
    std::vector<InputShape> shapes;
    Attributes attributes;

    ElementType model_type;
    std::tie(
        shapes,
        attributes.score_threshold,
        attributes.nms_threshold,
        attributes.max_delta_log_wh,
        attributes.num_classes,
        attributes.post_nms_count,
        attributes.max_detections_per_image,
        attributes.class_agnostic_box_regression,
        attributes.deltas_weights,
        model_type,
        targetDevice) = this->GetParam();

    if (model_type == element::f16)
        abs_threshold = 0.01;

    inType = outType = model_type;

    init_input_shapes(shapes);

    ov::ParameterVector params;
    for (auto&& shape : inputDynamicShapes) {
        params.push_back(std::make_shared<ov::op::v0::Parameter>(model_type, shape));
    }

    auto experimentalDetectron = std::make_shared<ov::op::v6::ExperimentalDetectronDetectionOutput>(
        params[0], // input_rois
        params[1], // input_deltas
        params[2], // input_scores
        params[3], // input_im_info
        attributes);
    function = std::make_shared<ov::Model>(
        ov::OutputVector{experimentalDetectron->output(0), experimentalDetectron->output(1)},
        "ExperimentalDetectronDetectionOutput");
}
} // namespace test
} // namespace ov
