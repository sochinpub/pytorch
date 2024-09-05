#include <ATen/ThreadLocalState.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

#include <c10/util/Logging.h>
#include <fmt/format.h>
#include <string_view>

#include <torch/csrc/distributed/c10d/PrefixStore.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupGloo.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupMPI.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupUCC.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupWrapper.hpp>

namespace c10d {

static ProcessGroup::BackendType strToBackendType(std::string_view backend) {
  if (backend == "undefined") {
    return ProcessGroup::BackendType::UNDEFINED;
  } else if (backend == "gloo") {
    return ProcessGroup::BackendType::GLOO;
  } else if (backend == "nccl") {
    return ProcessGroup::BackendType::NCCL;
  } else if (backend == "ucc") {
    return ProcessGroup::BackendType::UCC;
  } else if (backend == "mpi") {
    return ProcessGroup::BackendType::MPI;
  } else {
    return ProcessGroup::BackendType::CUSTOM;
  }
}
// 集合操作定义
std::string opTypeToString(OpType opType) {
  switch (opType) {
    case OpType::BROADCAST: // 0
      return "BROADCAST";
    case OpType::ALLREDUCE: // 1
      return "ALLREDUCE";
    case OpType::ALLREDUCE_COALESCED: // 2
      return "ALLREDUCE_COALESCED";
    case OpType::REDUCE:              // 3
      return "REDUCE";
    case OpType::ALLGATHER:           // 4
      return "ALLGATHER";
    case OpType::_ALLGATHER_BASE:     // 5
      return "_ALLGATHER_BASE";
    case OpType::ALLGATHER_COALESCED: // 6
      return "ALLGATHER_COALESCED";
    case OpType::GATHER:              // 7
      return "GATHER";
    case OpType::SCATTER:             // 8
      return "SCATTER";
    case OpType::REDUCE_SCATTER:      // 9
      return "REDUCE_SCATTER";
    case OpType::ALLTOALL_BASE:       // 10
      return "ALLTOALL_BASE";
    case OpType::ALLTOALL:            // 11
      return "ALLTOALL";
    case OpType::SEND:                // 12
      return "SEND";
    case OpType::RECV:                // 13
      return "RECV";
    case OpType::RECVANYSOURCE:       // 14
      return "RECVANYSOURCE";
    case OpType::BARRIER:             // 15
      return "BARRIER";
    case OpType::UNKNOWN:             // 100
      return "UNKNOWN";
    case OpType::_REDUCE_SCATTER_BASE: // 16
      return "_REDUCE_SCATTER_BASE";
    case OpType::COALESCED:            // 17
      return "COALESCED";
    case OpType::_ALLREDUCE_SPARSE:   // 18
      return "_ALLREDUCE_SPARSE";
    default:
      TORCH_INTERNAL_ASSERT(false, "Unknown op type!");
  }
  return "UNKNOWN";
}
// P2P 操作
bool isP2POp(OpType opType, bool batchP2P /*= false*/) {
  if (batchP2P)
    return false;
  return opType == OpType::SEND || opType == OpType::RECV ||
      opType == OpType::RECVANYSOURCE;
}

c10::intrusive_ptr<Backend> ProcessGroup::getBackend(
    c10::DeviceType deviceType) {
  // If there is a backend associated with this device type then return it
  // 设备类型关联了一个backend,直接返回
  // Sochin: 当前设备类型究竟是什么 ???
  if (deviceTypeToBackend_.find(deviceType) != deviceTypeToBackend_.end()) {
    return deviceTypeToBackend_.at(deviceType);
  }

  // Get the backend type associated with the device
  ProcessGroup::BackendType backendType{ProcessGroup::BackendType::UNDEFINED};
  try {
    // Sochin: TODO support MLU，这里是没有经过注册的
    backendType = deviceTypeToBackendType_.at(deviceType);
  } catch (const std::out_of_range& e) {
    TORCH_CHECK(
        false, "No backend type associated with device type ", deviceType);
  }

  // Check if the backend has already been initialized 是否完成了初始化
  if (backendTypeToBackend_.find(backendType) != backendTypeToBackend_.end()) {
    auto backend = backendTypeToBackend_.at(backendType);
    deviceTypeToBackend_[deviceType] = backend;
    return backend;
  }

  TORCH_CHECK(
      false,
      "Could not retrieve or create the backend ",
      backendType,
      " for device type ",
      deviceType);
}

ProcessGroup::ProcessGroup(
    const c10::intrusive_ptr<::c10d::Store>& store,
    int rank,
    int size,
    c10::intrusive_ptr<Options> options)
    : store_(store),
      rank_(rank),
      size_(size),
      options_(std::move(options)),
      backendType_(strToBackendType(options_->backend)),
      dist_debug_level_(debug_level()) {
  C10_LOG_API_USAGE_ONCE("c10d.process_group");
}
// 默认BackendType是未定义的
ProcessGroup::ProcessGroup(int rank, int size)
    : rank_(rank), size_(size), backendType_(BackendType::UNDEFINED) {}

ProcessGroup::~ProcessGroup() = default;

void ProcessGroup::init() {
  C10_LOG_API_USAGE_ONCE(
      fmt::format("c10d.process_group_{}", getBackendName()));
}

const std::string& ProcessGroup::getGroupName() const {
  TORCH_CHECK(!deviceTypeToBackend_.empty(), "ProcessGroup name not set");
  return deviceTypeToBackend_.begin()->second->getGroupName();
}

void ProcessGroup::setGroupName(const std::string& name) {
  for (auto& kv : deviceTypeToBackend_) {
    kv.second->setGroupName(name);
  }
}

const std::string& ProcessGroup::getGroupDesc() const {
  return pg_desc_;
}

void ProcessGroup::setGroupDesc(const std::string& name) {
  pg_desc_ = name;
  // Also set the group desc for all backends
  for (auto& kv : deviceTypeToBackend_) {
    kv.second->setGroupDesc(name);
  }
}

void ProcessGroup::enableCollectivesTiming() {
  for (auto& kv : deviceTypeToBackend_) {
    kv.second->enableCollectivesTiming();
  }
}

void ProcessGroup::release_resources() {
  store_.reset();
  deviceTypeToBackend_.clear();
  backendTypeToBackend_.clear();
}

} // namespace c10d
