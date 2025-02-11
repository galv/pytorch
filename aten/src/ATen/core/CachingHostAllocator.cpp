#include <ATen/core/CachingHostAllocator.h>

#include <array>

namespace at {

namespace {
  static std::array<std::function<HostAllocator*()>, at::COMPILE_TIME_MAX_DEVICE_TYPES>
    allocator_array{};
static std::array<uint8_t, at::COMPILE_TIME_MAX_DEVICE_TYPES>
    allocator_priority{};

} // anonymous namespace

void setHostAllocator(
    at::DeviceType device_type,
    std::function<at::HostAllocator*()> allocator,
    uint8_t priority) {
  if (priority >= allocator_priority[static_cast<int>(device_type)]) {
    allocator_array[static_cast<int>(device_type)] = std::move(allocator);
    allocator_priority[static_cast<int>(device_type)] = priority;
  }
}

at::HostAllocator* getHostAllocator(at::DeviceType device_type) {
  std::function<HostAllocator*()> allocator_thunk = allocator_array[static_cast<int>(device_type)];
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      bool(allocator_thunk), "Host Allocator for ", device_type, " is not set.");
  return allocator_thunk();
}

} // namespace at
