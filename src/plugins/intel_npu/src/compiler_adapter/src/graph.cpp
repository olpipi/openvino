// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "graph.hpp"

#include <iterator>

#ifdef __linux__
#    include <fcntl.h>
#    include <sys/mman.h>
#    include <unistd.h>

#    include <algorithm>
#    include <atomic>
#    include <cinttypes>
#    include <cstdio>
#    include <fstream>
#    include <string>
#    include <thread>
#    include <vector>
#endif

#include "intel_npu/config/options.hpp"
#include "intel_npu/utils/utils.hpp"
#include "intel_npu/utils/zero/zero_cmd_queue_pool.hpp"
#include "intel_npu/utils/zero/zero_utils.hpp"

namespace {

#ifdef __linux__
/**
 * @brief Reads the file behind a file-mapped blob into the page cache with pread() from several threads, skipping
 * pages that are already cached. Stops and joins on destruction.
 * @details Faulting the pages in through the mapping (ov::MappedMemory::hint_prefetch) does not progress while the
 * driver maps its device buffer during graph initialization, reading through the file descriptor does.
 */
class BlobFilePrefetch {
public:
    BlobFilePrefetch(const void* data, size_t size) {
        if (size < MIN_SIZE) {
            return;
        }
        const auto address = reinterpret_cast<uintptr_t>(data);
        std::ifstream maps("/proc/self/maps");
        for (std::string line; std::getline(maps, line);) {
            uintptr_t low = 0, high = 0;
            unsigned long long offset = 0;
            char path[4096] = {};
            const int fields = std::sscanf(line.c_str(),
                                           "%" SCNxPTR "-%" SCNxPTR " %*s %llx %*s %*s %4095s",
                                           &low,
                                           &high,
                                           &offset,
                                           path);
            if (fields < 3 || address < low || address >= high) {
                continue;
            }
            if (fields < 4 || path[0] != '/' || (_fd = ::open(path, O_RDONLY | O_CLOEXEC)) < 0) {
                return;  // not backed by a regular file
            }
            _address = address;
            _begin = static_cast<size_t>(offset) + (address - low);
            _end = _begin + size;
            _next = _begin;
            try {
                for (size_t i = 0; i < THREADS; ++i) {
                    _threads.emplace_back([this] {
                        run();
                    });
                }
            } catch (...) {
                _stop = true;
            }
            return;
        }
    }

    ~BlobFilePrefetch() {
        _stop = true;
        for (auto& thread : _threads) {
            thread.join();
        }
        if (_fd >= 0) {
            ::close(_fd);
        }
    }

private:
    static constexpr size_t MIN_SIZE = 64 * 1024 * 1024;
    static constexpr size_t CHUNK = 16 * 1024 * 1024;
    static constexpr size_t THREADS = 8;

    void run() noexcept {
        try {
            std::vector<char> buffer(CHUNK);
            std::vector<unsigned char> residency;
            const auto page = static_cast<uintptr_t>(::sysconf(_SC_PAGESIZE));
            for (size_t position; !_stop && (position = _next.fetch_add(CHUNK)) < _end;) {
                const size_t size = std::min(CHUNK, _end - position);
                const uintptr_t first = (_address + (position - _begin)) & ~(page - 1);
                const size_t length = _address + (position - _begin) + size - first;
                residency.resize((length + page - 1) / page);
                if (::mincore(reinterpret_cast<void*>(first), length, residency.data()) == 0 &&
                    std::all_of(residency.begin(), residency.end(), [](unsigned char resident) {
                        return resident & 1;
                    })) {
                    continue;
                }
                for (size_t done = 0; done < size && !_stop;) {
                    const ssize_t count = ::pread(_fd, buffer.data(), size - done, position + done);
                    if (count <= 0) {
                        break;
                    }
                    done += static_cast<size_t>(count);
                }
            }
        } catch (...) {
            // Best effort: the driver reads the blob anyway.
        }
    }

    int _fd = -1;
    uintptr_t _address = 0;
    size_t _begin = 0;
    size_t _end = 0;
    std::atomic<size_t> _next{0};
    std::atomic<bool> _stop{false};
    std::vector<std::thread> _threads;
};
#endif

}  // namespace

namespace intel_npu {

Graph::Graph(const std::shared_ptr<ZeGraphExtWrappers>& zeGraphExt,
             const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct,
             const GraphDescriptor& graphDesc,
             NetworkMetadata metadata,
             std::optional<ov::Tensor> blob,
             const std::optional<std::string>& compatibilityDescriptor,
             const bool blobIsPersistent,
             const std::shared_ptr<IProfilingDecoder>& profilingDecoder)
    : IGraph(),
      _zeGraphExt(zeGraphExt),
      _zeroInitStruct(zeroInitStruct),
      _graphDesc(graphDesc),
      _metadata(std::move(metadata)),
      _blob(std::move(blob)),
      _compatibilityDescriptor(compatibilityDescriptor),
      _profilingDecoder(profilingDecoder),
      _blobIsPersistent(blobIsPersistent),
      _logger("Graph", Logger::global().level()) {}

const NetworkMetadata& Graph::get_metadata() const {
    return _metadata;
}

void Graph::update_network_name(std::string_view name) {
    _metadata.name = name;
}

CommandQueueDesc Graph::get_command_queue_desc() const {
    std::lock_guard<std::mutex> lock(_commandQueueDescMutex);
    return _commandQueueDesc;
}

void Graph::set_workload_type(const ov::WorkloadType workloadType) {
    OPENVINO_ASSERT(_zeroInitStruct, "Driver is not initialized");

    std::lock_guard<std::mutex> lock(_commandQueueDescMutex);
    auto zeWorkloadType = zeroUtils::toZeQueueWorkloadType(workloadType);

    // If the graph already holds a command queue, that queue is the one used for execution,
    // so the workload type must be applied directly on it.
    if (_commandQueue && zeWorkloadType.has_value()) {
        // When shared common queue is disabled, workload type is set per command queue.
        // Update the existing queue if it has already been created.
        _commandQueue->setWorkloadType(zeWorkloadType.value());
        _workloadType = workloadType;
        return;
    }

    // Otherwise, the command queue is not created yet. Update the command queue descriptor,
    // since it will be used to create the command queue later.
    _commandQueueDesc.setWorkload(zeWorkloadType);
}

void Graph::set_model_priority(const ov::hint::Priority modelPriority) {
    OPENVINO_ASSERT(_zeroInitStruct, "Driver is not initialized");

    std::lock_guard<std::mutex> lock(_commandQueueDescMutex);
    auto zeModelPriority = zeroUtils::toZeQueuePriority(modelPriority);

    // If the graph already holds a command queue, that queue is the one used for execution,
    // so the priority must be applied on it (directly, or by recreating it on legacy drivers).
    if (_commandQueue) {
        if (_zeroInitStruct->isCommandQueueSetPrioritySupported()) {
            _commandQueue->setPriority(zeModelPriority);
            return;
        }

        // Legacy behavior: Create new command queue if setPriority is not supported.
        // When shared common queue is disabled, workload type is set per command queue.
        // Recreate the queue with the new priority while preserving the current workload type.
        if (_workloadType.has_value()) {
            auto zeWorkloadType = zeroUtils::toZeQueueWorkloadType(_workloadType.value());
            _commandQueueDesc.setWorkload(zeWorkloadType);
            _workloadType = std::nullopt;  // Clear the cached workload type after applying it to the new queue
        }

        _commandQueueDesc.setPriority(zeModelPriority);
        _commandQueue = ZeroCmdQueuePool::getInstance().getCommandQueue(_zeroInitStruct, _commandQueueDesc);
    }

    // Otherwise, the command queue is not created yet. Update the command queue descriptor,
    // since it will be used to create the command queue later.
    _commandQueueDesc.setPriority(zeModelPriority);
}

void* Graph::get_handle() const {
    return _graphDesc._handle;
}

std::pair<uint64_t, std::optional<std::vector<uint64_t>>> Graph::export_blob(std::ostream& stream) const {
    const uint8_t* blobPtr = nullptr;
    size_t blobSize;
    std::vector<uint8_t> blobVec;  // plugin needs to keep a copy of the blob for older drivers

    if (_blobIsReleased) {
        OPENVINO_THROW("Model was imported and released after initialization. Model export is not allowed anymore.");
    }

    if (_blob == std::nullopt) {
        OPENVINO_ASSERT(_zeGraphExt != nullptr, "Zero compiler adapter wasn't initialized");
        // when compiling the model using Compiler in Driver, the blob is handled by the driver
        _zeGraphExt->getGraphBinary(_graphDesc, blobVec, blobPtr, blobSize);
    } else {  // in all other cases, the blob is handled by the plugin
        blobPtr = static_cast<const uint8_t*>(_blob->data());
        blobSize = _blob->get_byte_size();
    }

    if (blobSize > static_cast<decltype(blobSize)>(std::numeric_limits<std::streamsize>::max())) {
        OPENVINO_THROW("Blob size is too large to be represented on a std::streamsize!");
    }
    stream.write(reinterpret_cast<const char*>(blobPtr), static_cast<std::streamsize>(blobSize));

    if (!stream) {
        _logger.error("Write blob to stream failed. Blob is broken!");
        return std::make_pair(0, std::nullopt);
    }

    if (_logger.level() >= ov::log::Level::INFO) {
        std::uint32_t result = 1171117u;
        for (const uint8_t* it = blobPtr; it != blobPtr + blobSize; ++it) {
            result = ((result << 7) + result) + static_cast<uint32_t>(*it);
        }
        _logger.info("Blob size: %zu, hash: %x", blobSize, result);
    }

    size_t size = utils::align_size_to_standard_page_size(blobSize);
    size_t paddingSize = size - blobSize;
    if (paddingSize > 0) {
        std::fill_n(std::ostream_iterator<char>(stream), paddingSize, 0);

        if (!stream) {
            _logger.error("Write padding to stream failed. Blob is broken!");
            return std::make_pair(0, std::nullopt);
        }

        _logger.info("Blob size with padding: %zu", size);
    }

    _logger.info("Write blob to stream successfully.");
    return std::make_pair(size, std::nullopt);
}

std::vector<ov::ProfilingInfo> Graph::process_profiling_output(const std::vector<uint8_t>& profData) const {
    OPENVINO_ASSERT(_profilingDecoder != nullptr, "No VCL profiling decoder is available for this graph");
    OPENVINO_ASSERT(_blob.has_value(), "VCL profiling decoder requires the compiled network blob");
    return _profilingDecoder->decode(profData, _blob.value());
}

bool Graph::has_profiling_decoder() const {
    return _profilingDecoder != nullptr;
}

void Graph::set_argument_value(uint32_t id, const void* data) const {
    if (_zeGraphExt == nullptr) {
        OPENVINO_THROW("Zero compiler adapter wasn't initialized");
    }
    _zeGraphExt->setGraphArgumentValue(_graphDesc, id, data);
}

void Graph::set_argument_value_with_strides(uint32_t id, const void* data, const std::vector<size_t>& strides) const {
    if (_zeGraphExt == nullptr) {
        OPENVINO_THROW("Zero compiler adapter wasn't initialized");
    }
    _zeGraphExt->setGraphArgumentValueWithStrides(_graphDesc, id, data, strides);
}

void Graph::initialize_impl(const Config& config) {
    _logger.debug("Graph initialize start");

    if (_zeGraphExt == nullptr || _graphDesc._handle == nullptr || _zeroInitStruct == nullptr) {
        // To ensure that no issues are thrown during subsequent calls.
        return;
    }

    bool sharedCommonQueue = config.get<SHARED_COMMON_QUEUE>();
    uint32_t commandQueueOptions = 0;
    if (config.get<TURBO>() && _zeroInitStruct->getCommandQueueDdiTable().version() >= ZE_MAKE_VERSION(1, 0)) {
        _logger.debug("Set ZE_NPU_COMMAND_QUEUE_OPTION_TURBO in command queue options");
        commandQueueOptions = commandQueueOptions | ZE_NPU_COMMAND_QUEUE_OPTION_TURBO;
    }
    if (config.get<RUN_INFERENCES_SEQUENTIALLY>()) {
        if (_zeroInitStruct->getCommandQueueDdiTable().version() >= ZE_MAKE_VERSION(1, 1)) {
            _logger.debug("Set ZE_NPU_COMMAND_QUEUE_OPTION_DEVICE_SYNC in command queue options");
            commandQueueOptions = commandQueueOptions | ZE_NPU_COMMAND_QUEUE_OPTION_DEVICE_SYNC;
        } else {
            OPENVINO_ASSERT(!sharedCommonQueue,
                            "RUN_INFERENCES_SEQUENTIALLY requires a command queue with device-sync support when "
                            "SHARED_COMMON_QUEUE is enabled");
        }
    }

    {
        std::lock_guard<std::mutex> lock(_commandQueueDescMutex);
        _commandQueueDesc = CommandQueueDesc{
            zeroUtils::toZeQueuePriority(config.get<MODEL_PRIORITY>()),
            config.has<WORKLOAD_TYPE>() ? zeroUtils::toZeQueueWorkloadType(config.get<WORKLOAD_TYPE>()) : std::nullopt,
            commandQueueOptions,
            this,
            sharedCommonQueue,
        };

        if (sharedCommonQueue == false) {
            // Keep it alive per compiled model when the shared common queue feature is disabled.
            _commandQueue = ZeroCmdQueuePool::getInstance().getCommandQueue(_zeroInitStruct, _commandQueueDesc);
        }
    }

    {
#ifdef __linux__
        // The driver copies the blob in initializeGraph(), read it from disk meanwhile.
        std::optional<BlobFilePrefetch> prefetch;
        if (_blob.has_value()) {
            const ov::Tensor& blob = *_blob;
            prefetch.emplace(blob.data(), blob.get_byte_size());
        }
#endif
        _zeGraphExt->initializeGraph(_graphDesc);
    }
    _logger.debug("Graph initialize finish");

    //  We are allowed to release the original blob because weights were loaded in NPU memory during
    //  _zeGraphExt->initializeGraph(). The driver will not access the original blob from this moment on, so we are
    //  releasing it here to avoid unnecessary memory usage.
    _blobIsReleased = release_blob(config);

    if (!_batchSize.has_value()) {
        _batchSize = determine_batch_size();
    }

    if (config.get<RUN_INFERENCES_SEQUENTIALLY>() &&
        _zeroInitStruct->getCommandQueueDdiTable().version() < ZE_MAKE_VERSION(1, 1)) {
        auto numberOfCommandLists = _batchSize.has_value() ? *_batchSize : 1;

        _lastSubmittedEvent.resize(numberOfCommandLists);
    }
    // To ensure that the initialization of the graph does not exit prematurely due to nullptrs
    _init_completed.store(true, std::memory_order_release);
}

bool Graph::release_blob(const Config& config) {
    if (_blobIsPersistent || _blob == std::nullopt || config.get<PERF_COUNT>() || _zeGraphExt == nullptr ||
        _zeGraphExt->isBlobDataImported(_graphDesc) || !_zeGraphExt->isInitStageRequired(_graphDesc)) {
        return false;
    }

    _blob = std::nullopt;
    _logger.debug("Blob is released");

    return true;
};

void Graph::set_last_submitted_event(const std::shared_ptr<Event>& event, size_t indexOfCommandList) {
    _lastSubmittedEvent[indexOfCommandList] = event;
}

const std::shared_ptr<Event>& Graph::get_last_submitted_event(size_t indexOfCommandList) const {
    return _lastSubmittedEvent[indexOfCommandList];
}

void Graph::resize_last_submitted_event(size_t batch) {
    _lastSubmittedEvent.resize(batch);
}

void Graph::set_batch_size(std::size_t batch) {
    _batchSize = batch;
}

uint32_t Graph::get_unique_id() {
    return _uniqueId++;
}

void Graph::set_last_submitted_id(uint32_t id_index) {
    _lastSubmittedId = id_index;
}

uint32_t Graph::get_last_submitted_id() const {
    return _lastSubmittedId;
}

std::optional<std::string_view> Graph::get_compatibility_descriptor() const {
    return _compatibilityDescriptor;
}

std::optional<bool> Graph::is_profiling_blob() const {
    std::optional<bool> profilingEnabled;

    if (_zeGraphExt != nullptr) {
        profilingEnabled = _zeGraphExt->isProfilingEnabled(_graphDesc);
    }

    if (!profilingEnabled.has_value()) {
        _logger.debug("Cannot determine if the blob was compiled for profiling");
    }

    return profilingEnabled;
}

std::optional<size_t> Graph::determine_batch_size() {
    if (!_metadata.outputs.at(0).shapeFromIRModel.has_value()) {
        _logger.debug("Batching on the plugin is not used, batching is handled by the compiler");
        return std::nullopt;
    }

    const ov::PartialShape& firstShape = *_metadata.outputs.at(0).shapeFromIRModel;
    if (firstShape.is_dynamic() || firstShape.rank().get_length() == 0) {
        return std::nullopt;
    }

    const size_t candidateBatchSize = firstShape[utils::BATCH_AXIS].get_max_length();
    if (candidateBatchSize == 0 || candidateBatchSize == utils::DEFAULT_BATCH_SIZE) {
        _logger.debug("Batching on the plugin is not used, batching is handled by the compiler");
        return std::nullopt;
    }

    auto checkDescriptorsUseCandidateBatchSize = [candidateBatchSize](const std::vector<IODescriptor>& descriptors) {
        for (const IODescriptor& descriptor : descriptors) {
            OPENVINO_ASSERT(descriptor.shapeFromIRModel.has_value(),
                            "Missing value for the \"shapeFromIRModel\" attribute, I/O descriptor");

            const ov::PartialShape& shapeFromCompiler = descriptor.shapeFromCompiler;
            const ov::PartialShape& shapeFromIRModel = *descriptor.shapeFromIRModel;

            if (shapeFromCompiler.is_dynamic() || shapeFromCompiler.rank().get_length() == 0 ||
                *shapeFromCompiler.begin() != utils::DEFAULT_BATCH_SIZE) {
                return false;
            }

            if (!descriptor.isStateInput && !descriptor.isStateOutput && !descriptor.isShapeTensor) {
                if (shapeFromIRModel.is_dynamic() || shapeFromIRModel.rank().get_length() == 0 ||
                    *shapeFromIRModel.begin() != candidateBatchSize) {
                    return false;
                }
            }
        }

        return true;
    };

    if (!checkDescriptorsUseCandidateBatchSize(_metadata.inputs) ||
        !checkDescriptorsUseCandidateBatchSize(_metadata.outputs)) {
        _logger.debug("Batching on the plugin is not used, batching is handled by the compiler");
        return std::nullopt;
    }

    _logger.debug("Batching is handled by the plugin");

    return candidateBatchSize;
}

const std::optional<std::size_t> Graph::get_batch_size() const {
    return _batchSize;
}

void Graph::evict_memory() {
    if (_zeGraphExt != nullptr) {
        _zeGraphExt->evict_memory(_graphDesc);
    }
}

Graph::~Graph() {
    // make sure all the context-dependent components are destroyed before the zero context is destroyed
    if (_zeGraphExt != nullptr) {
        _zeGraphExt->destroyGraph(_graphDesc);
    }

    if (!_lastSubmittedEvent.empty()) {
        _lastSubmittedEvent.clear();
    }
}

}  // namespace intel_npu
