#pragma once

#include <ATen/core/CachingHostAllocator.h>
#include <c10/core/Allocator.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Deprecated.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>

namespace at::cuda {

// TODO: Should I use a pimpl in order to hide these sorts of details?

// Maybe I should really use a pointer-to-implementation design to avoid exposing this.
struct MempoolIdHash {
  std::size_t operator()(const MempoolId_t& mempool_id) const noexcept {
    return mempool_id.first != 0 ? mempool_id.first : mempool_id.second;
  }
};



struct GraphBlock;
struct PrivatePool;

/**
 * CUDAStreamCapturableCachingHostAllocator is used only during stream
 * capture.  It is accessed from CUDAGraph.cpp via the functions in
 * CachingHostAllocator.h.  In particular,
 * getCUDACachingHostAllocator() will select the singleton
 * CUDAStreamCapturableCachingHostAllocator whenever the current
 * stream is doing stream capture.
 *
 * Unlike CUDACachingHostAllocator,
 * CUDAStreamCapturableCachingHostAllocator cannot query events
 * associated with usages to detect when a host allocation is free to
 * be reused. This is because cuda events will never actually be
 * recorded during stream capture. They will be recorded only during
 * graph replay. Therefore, during stream capture to a graph, a naive
 * implementation based on the original design would simply fail to
 * reuse any allocations whatsoever, within a cuda graph, which is not
 * acceptable for a caching allocator.
 *
 * This implementation uses a simple algorithm to decide whether a
 * particular host allocation block can be recycled. At every instance
 * of record_event(), it will insert an empty node into the currently
 * capturing stream. If (1) the reference count of a cudaHostAlloc()
 * created block has gone to 0 (tracked via the `allocated` field in
 * HostBlock), and (2) there is a path from every empty node created
 * by a call to record_event() to the current node in stream capture,
 * then this block can be reused, and is therefore moved to the free
 * list.
 *
 * TODO: Consider how to make this work in the case where a memory
 * pool is shared across several cuda graphs. Especially when we have
 * external events, which break the assumption in pytorch that cuda
 * graphs are "atomic" units of execution. We probably need to treat
 * the insertion of an external event into the graph as a possible
 * "usage" with an indeterminate end point, which isn't great because
 * it is very pessimizing.
 *
 * External events PR: https://github.com/pytorch/pytorch/pull/146145
 */
struct CUDAStreamCapturableCachingHostAllocator final : public HostAllocator {
  CUDAStreamCapturableCachingHostAllocator() = default;

  at::DataPtr allocate(size_t size) override;
  void free(void* ctx);

  bool record_event(void* ptr, void* ctx, c10::Stream stream) override;

  void empty_cache() override;

  void copy_data(void* dest [[maybe_unused]], const void* src [[maybe_unused]], std::size_t count [[maybe_unused]]) const override;

  HostStats get_stats() override;

  void reset_accumulated_stats() override;

  void reset_peak_stats() override;

  void begin_allocate_to_pool(std::pair<unsigned long long, unsigned long long> pool_id, std::function<bool(c10::Stream)> filter);

  void release_pool(std::pair<unsigned long long, unsigned long long> pool_id);
  void end_allocate_to_pool(std::pair<unsigned long long, unsigned long long> pool_id);
private:
  size_t size_index(size_t size);

  GraphBlock* get_free_block_under_lock(PrivatePool& cuda_mem_pool, size_t size);

  void free_finished_allocations_under_lock(PrivatePool& cuda_mem_pool);
  // Unlike CUDACachingHostAllocator,
  // CUDAStreamCapturableCachingHostAllocator uses only a single
  // mutex. There are a few reasons:

  // 1. simplicity. Stream capture rarely spans multiple threads in
  // pytorch. The main situation in which this happens (that I know
  // of) is when you do stream capture over both forward and backward
  // pass, and did not do
  // torch.autograd.grad_mode.set_multithreading_enabled(False)
  // 2. CUDAStreamCapturableCachingHostAllocator is used only during stream capture,
  // which is allowed to be "slow" since it happens only once (this is debatable).
  // 3. The only expensive API call we do is cudaHostAlloc(). Events are no longer used.
  alignas(64) std::mutex m_;
  std::vector<std::pair<MempoolId_t, std::function<bool(CUDAStream)>>> captures_underway_;
  ska::flat_hash_map<MempoolId_t, std::unique_ptr<PrivatePool>, MempoolIdHash> cuda_mem_pools_;
};

TORCH_CUDA_CPP_API at::cuda::CUDAStreamCapturableCachingHostAllocator* getCUDAStreamCapturableCachingHostAllocator();

//
// A caching allocator for CUDA host allocations (pinned memory).
//
// This provides a drop-in replacement for THCudaHostAllocator, which re-uses
// freed pinned (page-locked) memory allocations. This avoids device
// synchronizations due to cudaFreeHost calls.
//
// To ensure correct behavior, THCCachingHostAllocator_recordEvent must be
// called anytime a pointer from this allocator is used in a cudaMemcpyAsync
// call between host and device, and passed the corresponding context from the
// allocation. This is currently invoked by at::native::copy_kernel_cuda.
//
C10_DEPRECATED_MESSAGE(
  "at::cuda::getCachingHostAllocator() is deprecated. Please use at::getHostAllocator(at::kCUDA) instead.")
inline TORCH_CUDA_CPP_API at::HostAllocator* getCachingHostAllocator() {
  return at::getHostAllocator(at::kCUDA);
}

// Records an event in the specified stream. The allocation corresponding to the
// input `ptr`/`ctx` will not be re-used until the event has occurred.
C10_DEPRECATED_MESSAGE(
  "at::cuda::CachingHostAllocator_recordEvent(...) is deprecated. Please use at::getHostAllocator(at::kCUDA)->record_event(...) instead.")
inline TORCH_CUDA_CPP_API bool CachingHostAllocator_recordEvent(
    void* ptr,
    void* ctx,
    c10::cuda::CUDAStream stream) {
  return getHostAllocator(at::kCUDA)->record_event(ptr, ctx, stream.unwrap());
}

// Releases cached pinned memory allocations via cudaHostFree
C10_DEPRECATED_MESSAGE(
  "at::cuda::CachingHostAllocator_emptyCache() is deprecated. Please use at::getHostAllocator(at::kCUDA)->empty_cache() instead.")
inline TORCH_CUDA_CPP_API void CachingHostAllocator_emptyCache() {
  getHostAllocator(at::kCUDA)->empty_cache();
}

C10_DEPRECATED_MESSAGE(
  "at::cuda::HostAlloc(...) is deprecated. Please use at::getHostAllocator(at::kCUDA)->allocate(...) instead.")
inline TORCH_CUDA_CPP_API at::DataPtr HostAlloc(size_t size) {
  return getHostAllocator(at::kCUDA)->allocate(size);
}

C10_DEPRECATED_MESSAGE(
  "at::cuda::CachingHostAllocator_getStats() is deprecated. Please use at::getHostAllocator(at::kCUDA)->get_stats() instead.")
inline TORCH_CUDA_CPP_API at::HostStats CachingHostAllocator_getStats() {
  return getHostAllocator(at::kCUDA)->get_stats();
}

C10_DEPRECATED_MESSAGE(
  "at::cuda::CachingHostAllocator_resetAccumulatedStats() is deprecated. Please use at::getHostAllocator(at::kCUDA)->reset_accumulated_stats() instead.")
inline TORCH_CUDA_CPP_API void CachingHostAllocator_resetAccumulatedStats() {
  getHostAllocator(at::kCUDA)->reset_accumulated_stats();
}

C10_DEPRECATED_MESSAGE(
  "at::cuda::CachingHostAllocator_resetPeakStats() is deprecated. Please use at::getHostAllocator(at::kCUDA)->reset_peak_stats() instead.")
inline TORCH_CUDA_CPP_API void CachingHostAllocator_resetPeakStats() {
  getHostAllocator(at::kCUDA)->reset_peak_stats();
}

} // namespace at::cuda
