// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functional_test_utils/core_config.hpp"

void CoreConfiguration(LayerTestsUtils::LayerTestsCommon* test) {
}


void CoreConfiguration(LayerTestsUtilsNew::LayerTestsCommon* test) {
    // std::shared_ptr<ov::Core> core = ov::test::utils::PluginCache::get().core();
    // auto availableDevices = core->get_available_devices();
    // std::string targetDevice = std::string(ov::test::conformance::targetDevice);
    // if (std::find(availableDevices.begin(), availableDevices.end(), targetDevice) == availableDevices.end()) {
    //     core->register_plugin(ov::util::make_plugin_library_name(CommonTestUtils::getExecutableDirectory(),
    //                                                             std::string(ov::test::conformance::targetPluginName) + IE_BUILD_POSTFIX),
    //                          ov::test::conformance::targetDevice);
    // }
}
