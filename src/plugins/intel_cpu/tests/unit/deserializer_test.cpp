// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <pugixml.hpp>

#include "openvino/core/except.hpp"
#include "openvino/core/rt_info/weightless_caching_attributes.hpp"
#include "openvino/runtime/aligned_buffer.hpp"
#include "utils/graph_serializer/deserializer.hpp"

namespace ov::intel_cpu {

// Exposes the protected method under test without going through the full cache blob format.
class TestableXmlDeserializer : public XmlDeserializer {
public:
    using XmlDeserializer::XmlDeserializer;
    using XmlDeserializer::set_constant_num_buffer;
};

}  // namespace ov::intel_cpu

// A weightless-cache Const whose <data shape="..."> makes shape_size(shape) * bitwidth of the
// original on-disk dtype overflow size_t must still be rejected in the dtype-conversion path,
// not let a too-small origin weights buffer pass validation.
TEST(CpuXmlDeserializerTest, weightless_dtype_conversion_shape_overflow_throws) {
    pugi::xml_document doc;
    auto layer = doc.append_child("layer");
    layer.append_attribute("type").set_value("Const");

    auto data = layer.append_child("data");
    data.append_attribute("element_type").set_value("f32");
    data.append_attribute("shape").set_value("4611686018427387904");  // 2^62

    auto rt_info = layer.append_child("rt_info");
    auto attribute = rt_info.append_child("attribute");
    attribute.append_attribute("name").set_value(ov::WeightlessCacheAttribute::get_type_info_static().name);
    attribute.append_attribute("original_size").set_value(uint64_t{1});
    attribute.append_attribute("bin_offset").set_value(uint64_t{0});
    attribute.append_attribute("original_dtype").set_value("u4");

    auto origin_weights = std::make_shared<ov::AlignedBuffer>(1);

    std::unordered_map<std::string, ov::OpSet> opsets;
    std::unordered_map<ov::DiscreteTypeInfo, ov::BaseOpExtension::Ptr> extensions;
    std::unordered_map<std::string, std::shared_ptr<ov::op::util::Variable>> variables;

    ov::intel_cpu::TestableXmlDeserializer deserializer(layer,
                                                        nullptr,
                                                        origin_weights,
                                                        opsets,
                                                        extensions,
                                                        variables,
                                                        11);

    std::shared_ptr<ov::AlignedBuffer> result;
    ov::AttributeAdapter<std::shared_ptr<ov::AlignedBuffer>> adapter(result);

    EXPECT_THROW(deserializer.set_constant_num_buffer(adapter), ov::Exception);
}
