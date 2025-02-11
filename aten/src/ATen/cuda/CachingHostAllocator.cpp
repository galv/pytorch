#include <ATen/cuda/CachingHostAllocator.h>

#include <ATen/DeviceGuard.h>
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/detail/CUDAHooks.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <c10/core/thread_pool.h>
#include <c10/cuda/CUDAAllocatorConfig.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>

#include <c10/util/flat_hash_map.h>

#include <cuda_runtime_api.h>
#include <future>
#include <unordered_map>
#include <vector>

namespace at::cuda {
namespace {

// Note: cudaEventCreate when concurrently invoked from multiple threads can be
// very expensive (at least on certain device/driver combinations). Thus, we a)
// serialize event creation at a per-device level, and b) pool the events to
// avoid constantly calling cudaEventCreate/cudaEventDestroy. This results in
// significant improvements in multithreaded workloads with high allocation
// rates.
class EventPool {
 public:
  using Event = std::unique_ptr<
      at::cuda::CUDAEvent,
      std::function<void(at::cuda::CUDAEvent*)>>;
  EventPool() : pools_(at::cuda::device_count()) {}

  Event get(DeviceIndex device) {
    TORCH_INTERNAL_ASSERT(0 <= device);
    TORCH_INTERNAL_ASSERT(device < static_cast<DeviceIndex>(pools_.size()));
    auto& pool = pools_[device];
    auto destructor = [&pool](at::cuda::CUDAEvent* event) {
      std::lock_guard<std::mutex> g(pool.mutex_);
      pool.event_pool_.push_back(std::unique_ptr<at::cuda::CUDAEvent>(event));
    };

    // Try to acquire an event from the per-device pool.
    {
      std::lock_guard<std::mutex> g(pool.mutex_);
      if (!pool.event_pool_.empty()) {
        auto* event = pool.event_pool_.back().release();
        pool.event_pool_.pop_back();
        return Event(event, destructor);
      }
    }
    // otherwise, allocate a new event that will be returned to the pool on
    // destruction.
    return Event(
        std::make_unique<at::cuda::CUDAEvent>(cudaEventDisableTiming).release(),
        destructor);
  }

  void empty_cache() {
    for (auto& pool : pools_) {
      std::lock_guard<std::mutex> g(pool.mutex_);
      pool.event_pool_.clear();
    }
  }

 private:
  struct PerDevicePool {
    alignas(64) std::mutex mutex_;
    std::vector<std::unique_ptr<at::cuda::CUDAEvent>> event_pool_;
  };
  std::vector<PerDevicePool> pools_;
};

using Block = HostBlock<CUDAStream>;

struct CUDACachingHostAllocatorImpl
    : public CachingHostAllocatorImpl<CUDAStream, EventPool::Event> {
 private:
  std::unordered_map<void*, bool> use_host_register;

  void allocate_host_memory(size_t size, void** ptr) override {
    // Pinned memory pointers allocated by any device can be directly used by
    // any other device, regardless of the current device at the time of
    // allocation, since we assume unified addressing. So we grab any existing
    // primary context, if available. See pytorch/pytorch#21081.
    // This can be a large performance hit if we cross NUMA nodes by allocating
    // and pinning memory on one side of the NUMA node and then using it on the
    // other side. Thankfully, we use one process per GPU, so we don't run into
    // this issue.
    at::OptionalDeviceGuard device_guard;
    auto primary_ctx_device_index =
        c10::cuda::getDeviceIndexWithPrimaryContext();
    if (primary_ctx_device_index.has_value()) {
      device_guard.reset_device(
          at::Device(at::DeviceType::CUDA, *primary_ctx_device_index));
    }

    auto start = std::chrono::steady_clock::now();
    bool use_register = c10::cuda::CUDACachingAllocator::CUDAAllocatorConfig::pinned_use_cuda_host_register();
    if (use_register) {
      allocWithCudaHostRegister(ptr, size);
    } else {
      // Use cudaHostAlloc for allocating pinned memory (global lock in driver)
      C10_CUDA_CHECK(cudaHostAlloc(ptr, size, cudaHostAllocDefault));
    }

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Update the statistics on the time spent on cudaHostAlloc/hostRegister
    {
      std::lock_guard<std::mutex> g(stats_.timing_mutex_);
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(use_host_register.count(*ptr) == 0);
      use_host_register[*ptr] = use_register;
      stats_.host_alloc_time.increase(duration.count());
    }
  }

  void free_block(Block* block) override {
    auto start = std::chrono::steady_clock::now();
    // Users may change the allocator config at will. torch unit tests do this.
    // However, allocations using cudaHostRegister should use corresonding
    // cudaHostUnregister and similarly for cudaHostAlloc / cudaFreeHost.
    void* ptr = block->ptr_;
    bool use_register = false;
    {
      std::lock_guard<std::mutex> g(stats_.timing_mutex_);
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(use_host_register.count(ptr) == 1);
      use_register = use_host_register[ptr];
    }
    if (use_register) {
      AT_CUDA_CHECK(cudaHostUnregister(ptr));
      // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
      std::free(ptr);
    } else {
      AT_CUDA_CHECK(cudaFreeHost(ptr));
    }
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Update the statistics on the time spent on cudaFreeHost/hostUnregister
    {
      std::lock_guard<std::mutex> g(stats_.timing_mutex_);
      use_host_register.erase(ptr);
      stats_.host_free_time.increase(duration.count());
    }
  }

  void record_stream(
      std::optional<std::vector<EventPool::Event>>& events,
      CUDAStream stream) override {
    auto event = create_event_internal(stream.device_index());
    event->record(stream);
    events->push_back(std::move(event));
  }

  bool query_event(EventPool::Event& event) override {
    cudaError_t err = cudaEventQuery(*event);
    if (err == cudaErrorNotReady) {
      (void)cudaGetLastError(); // clear CUDA error
      return false;
    } else if (err != cudaSuccess) {
      C10_CUDA_CHECK(err);
    }
    return true;
  }

  bool pinned_use_background_threads() override {
    return c10::CachingAllocator::AcceleratorAllocatorConfig::
        pinned_use_background_threads();
  }

  EventPool::Event create_event_internal(DeviceIndex idx) {
    // Leak the event pool to avoid shutdown issue.
    static auto* event_pool = new EventPool();
    return event_pool->get(idx);
  }

  TaskThreadPool* getThreadPool() {
    static TaskThreadPool* pool = new TaskThreadPool(
        static_cast<int>(c10::cuda::CUDACachingAllocator::CUDAAllocatorConfig::
            pinned_max_register_threads()));
    return pool;
  }

  void mapPagesForRegister(
      const void* ptr,
      size_t size,
      size_t i,
      size_t numThreads,
      size_t pageSize) {
    uintptr_t start = (uintptr_t)ptr + (size * i / numThreads);
    uintptr_t end = (uintptr_t)start + (size / numThreads);
    if (i == (numThreads - 1)) {
      end = (uintptr_t)ptr + size;
    }

    // pre-fault/map the pages by setting the first byte of the page
    uintptr_t alignedStart =
        (((uintptr_t)start + pageSize - 1) & ~(pageSize - 1));
    for (uintptr_t p = alignedStart; p < ((uintptr_t)end); p += pageSize) {
      // NOLINTNEXTLINE(performance-no-int-to-ptr)
      memset((void*)p, 0, 1);
    }
  }

  void allocWithCudaHostRegister(void** ptr, size_t roundSize) {
    // Here we do regular allocation, pre-fault/map the pages, and then do
    // cudaHostRegister with GPU mapping flags to lock the pages, so we
    // can minimize the cost for the cuda global lock.
    // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
    *ptr = std::malloc(roundSize);

    // Parallelize the mapping/registering of pages to reduce wall time
    size_t pageSize = (1 << 12); // 4kB pages
    size_t numMapThreads = c10::cuda::CUDACachingAllocator::
        CUDAAllocatorConfig::pinned_num_register_threads();
    if ((numMapThreads > 1) && (roundSize >= (pageSize * numMapThreads))) {
      // parallelize the mapping of pages with a threadpool
      auto* pool = getThreadPool();
      std::vector<std::promise<void>> promises;
      std::vector<std::future<void>> futures;
      promises.reserve(numMapThreads);
      futures.reserve(numMapThreads);

      for (size_t i = 0; i < numMapThreads; i++) {
        promises.emplace_back();
        futures.push_back(promises[i].get_future());
        auto task = [this,
                     i,
                     ptr,
                     roundSize,
                     numMapThreads,
                     pageSize,
                     &promises]() mutable {
          mapPagesForRegister(
              *ptr,
              roundSize,
              i, // thread task-id
              numMapThreads,
              pageSize);
          // set the promise when mapping pages are done
          promises[i].set_value();
        };
        pool->run(task);
      }
      for (auto& future : futures) {
        future.wait();
      }
    } else {
      // Map pages in the same thread
      mapPagesForRegister(*ptr, roundSize, 0, 1, pageSize);
    }

    // Register the mapped pages using cudaHostRegister
    AT_CUDA_CHECK(
        cudaHostRegister(*ptr, roundSize, cudaHostRegisterDefault));
  }
};

DECLARE_HOST_ALLOCATOR(
    CUDACachingHostAllocator,
    CUDACachingHostAllocatorImpl,
    raw_local_deleter,
    caching_host_allocator)

REGISTER_HOST_ALLOCATOR(at::kCUDA, &caching_host_allocator)
} // anonymous namespace

struct GraphBlock {
  GraphBlock(void* ptr_, std::size_t size_) : ptr(ptr_), size(size_) {}

  void *ptr;
  std::size_t size;
  std::vector<cudaGraphNode_t> using_nodes;
  bool allocated_{true};
};

struct PrivatePool {
  int use_count{1};
  ska::flat_hash_set<GraphBlock*> blocks;

  // We keep free list as a vector of free lists, one for each power of two
  // size. This allows us to quickly find a free block of the right size.
  // We use deque to store per size free list
  std::vector<std::deque<GraphBlock*>> free_list_ = std::vector<std::deque<GraphBlock*>>(MAX_SIZE_INDEX);
};

cudaGraphNode_t insert_empty_node_under_lock(GraphBlock* context, CUDAStream stream) {
    // N.B.: m_ must be held here in order to protect the cudaGraph_t
    // being populated by stream capture. While this class doesn't
    // have the currently capturing graph as a member, cudaGraph_t
    // objects are not thread-safe, so we must enforce mutual
    // exclusion when mutating it.
    // See: https://docs.nvidia.com/cuda/archive/12.9.1/cuda-runtime-api/graphs-thread-safety.html#graphs-thread-safety
    cudaStreamCaptureStatus status{};
    cudaGraph_t currently_capturing_graph{};
    const cudaGraphNode_t* dependencies{};
    size_t num_dependencies = 0;
    AT_CUDA_CHECK(cudaStreamGetCaptureInfo(stream,
                                               &status,
                                               nullptr,
                                               &currently_capturing_graph,
                                               &dependencies,
                                               &num_dependencies));
    TORCH_INTERNAL_ASSERT(status == cudaStreamCaptureStatusActive);

    // N.B.: Mutating the cudaGraph_t here is not thread safe.
    cudaGraphNode_t new_node{};
    AT_CUDA_CHECK(cudaGraphAddEmptyNode(
                                       &new_node,
                                       currently_capturing_graph,
                                       dependencies,
                                       num_dependencies));

    AT_CUDA_CHECK(cudaStreamUpdateCaptureDependencies(
                                                      stream, &new_node, 1, cudaStreamSetCaptureDependencies));

    if (context != nullptr) {
      context->using_nodes.push_back(new_node);
    }

    return new_node;
}

bool path_exists(cudaGraphNode_t source,
                 cudaGraphNode_t destination) {
    // Use an unordered_set to track visited nodes.
    std::unordered_set<cudaGraphNode_t> visited;

    // Define a recursive lambda to perform DFS.
    std::function<bool(cudaGraphNode_t)> dfs = [&](cudaGraphNode_t current) -> bool {
        if (current == destination) {
            return true;
        }
        visited.insert(current);

        // First, get the number of dependent nodes.
        size_t numDependent = 0;
        AT_CUDA_CHECK(cudaGraphNodeGetDependentNodes(current, nullptr, &numDependent));
        // Allocate a vector to hold the dependent nodes.
        std::vector<cudaGraphNode_t> dependents(numDependent);
        AT_CUDA_CHECK(cudaGraphNodeGetDependentNodes(current, dependents.data(), &numDependent));
        // Recursively search each dependent node.
        for (const auto& next : dependents) {
            if (visited.find(next) == visited.end()) {
                if (dfs(next)) {
                    return true;
                }
            }
        }
        return false;
    };

    return dfs(source);
}

CUDAStreamCapturableCachingHostAllocator stream_capturable_caching_host_allocator;

void stream_capturable_raw_local_deleter(void* ptr) {
  stream_capturable_caching_host_allocator.free(ptr);
}

  at::DataPtr CUDAStreamCapturableCachingHostAllocator::allocate(size_t size) {
    void* host_ptr = nullptr;
    CUDAStream stream = getCurrentCUDAStream();

    std::lock_guard<std::mutex> g(m_);

    for (auto &&[mempool_id, filter]: captures_underway_) {
      if (filter(stream)) {
        PrivatePool& cuda_mem_pool = *cuda_mem_pools_.at(mempool_id);

        size_t round_size = c10::llvm::PowerOf2Ceil(size);

        // TODO: Consider doing this only if there is no round_size block on the free list.
        // We don't want allocation to be O(N), where N is the number of active blocks.
        free_finished_allocations_under_lock(cuda_mem_pool);

        // First, try to allocate from the free list
        auto* block = get_free_block_under_lock(cuda_mem_pool, round_size);
        if (block) {
          cuda_mem_pool.blocks.insert(block);
          return at::DataPtr(block->ptr, block, stream_capturable_raw_local_deleter, DeviceType::CPU);
        } else {
          at::cuda::CUDAStreamCaptureModeGuard g{cudaStreamCaptureModeRelaxed};
          AT_CUDA_CHECK(cudaHostAlloc(&host_ptr, round_size, cudaHostAllocDefault));
          block = new GraphBlock(host_ptr, round_size);
          cuda_mem_pool.blocks.insert(block);
          return at::DataPtr(host_ptr, block, stream_capturable_raw_local_deleter, DeviceType::CPU);
        }
      }
    }

    TORCH_INTERNAL_ASSERT(false, "CUDAStreamCapturableCachingHostAllocator::allocate() is expected to be called only within a capturing context");
  }

  void CUDAStreamCapturableCachingHostAllocator::free(void* ctx) {
    // N.B.: I don't think the lock here is required
    std::lock_guard<std::mutex> g(m_);
    auto block = (GraphBlock*)ctx;
    block->allocated_ = false;
    // N.B.: We could eagerly try to add this to the free list now,
    // but we would need to keep the free_finished_allocations() call
    // in allocate() anyway since record_event may have been called,
    // which prevents immediate recycling. I suppose we could check if
    // block->using_nodes is empty, though.
  }

  bool CUDAStreamCapturableCachingHostAllocator::record_event(void* ptr, void* ctx, c10::Stream stream) {
    std::lock_guard<std::mutex> g(m_);
    insert_empty_node_under_lock((GraphBlock*)ctx, CUDAStream(stream));
    return true;
  }

  void CUDAStreamCapturableCachingHostAllocator::empty_cache() {
    // We cannot release any memory if it has been allocated within a cuda
    // graph, until that cuda graph gets destroyed. So this is a no-op
    // operation. release_pool does the actual freeing.
  }

  void CUDAStreamCapturableCachingHostAllocator::copy_data(void* dest [[maybe_unused]], const void* src [[maybe_unused]], std::size_t count [[maybe_unused]]) const {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "Not implemented for copy_data");
  }

  HostStats CUDAStreamCapturableCachingHostAllocator::get_stats() {TORCH_CHECK_NOT_IMPLEMENTED(false, "Not implemented for copy_data");}

  void CUDAStreamCapturableCachingHostAllocator::reset_accumulated_stats() {TORCH_CHECK_NOT_IMPLEMENTED(false, "Not implemented for copy_data");}

  void CUDAStreamCapturableCachingHostAllocator::reset_peak_stats() {TORCH_CHECK_NOT_IMPLEMENTED(false, "Not implemented for copy_data");}

  void CUDAStreamCapturableCachingHostAllocator::begin_allocate_to_pool(std::pair<unsigned long long, unsigned long long> pool_id, std::function<bool(c10::Stream)> filter) {
    std::lock_guard<std::mutex> g(m_);
    if (!cuda_mem_pools_.count(pool_id)) {
      cuda_mem_pools_.emplace(pool_id, std::make_unique<PrivatePool>());
    } else {
      PrivatePool& pool = *cuda_mem_pools_.at(pool_id);
      TORCH_INTERNAL_ASSERT(pool.use_count > 0);
      pool.use_count++;
    }
    captures_underway_.emplace_back(pool_id, std::move(filter));
  }

  void CUDAStreamCapturableCachingHostAllocator::release_pool(std::pair<unsigned long long, unsigned long long> pool_id) {
    std::lock_guard<std::mutex> g(m_);
    PrivatePool& pool = *cuda_mem_pools_.at(pool_id);
    int uc = --pool.use_count;

    if (uc == 0) {
      for (auto &&block: pool.blocks) {
        // TORCH_INTERNAL_ASSERT(block->allocated_); <- Incorrect. Suppose we free a block but then never try to allocate again. No time to move to move it to the free list.
        AT_CUDA_CHECK(cudaFreeHost(block->ptr));
        delete block;
      }

      for (auto &&free_list: pool.free_list_) {
        for (auto&& block: free_list) {
          TORCH_INTERNAL_ASSERT(!block->allocated_);
          AT_CUDA_CHECK(cudaFreeHost(block->ptr));
          delete block;
        }
      }
      cuda_mem_pools_.erase(pool_id);
    }
  }

  void CUDAStreamCapturableCachingHostAllocator::end_allocate_to_pool(std::pair<unsigned long long, unsigned long long> pool_id) {
    std::lock_guard<std::mutex> g(m_);
    free_finished_allocations_under_lock(*cuda_mem_pools_.at(pool_id));
    auto it = std::find_if(captures_underway_.begin(), captures_underway_.end(),
        [pool_id](const auto& elem) {
            return std::get<0>(elem) == pool_id;
        });

    TORCH_INTERNAL_ASSERT(it != captures_underway_.end());
    captures_underway_.erase(it);
  }

  size_t CUDAStreamCapturableCachingHostAllocator::size_index(size_t size) {
    return c10::llvm::Log2_64_Ceil(size);
  }

  GraphBlock* CUDAStreamCapturableCachingHostAllocator::get_free_block_under_lock(PrivatePool& cuda_mem_pool, size_t size) {
    // m_ must be held
    auto index = size_index(size);
    std::deque<GraphBlock*>& free_blocks = cuda_mem_pool.free_list_[index];
    if (!free_blocks.empty()) {
      GraphBlock* block = free_blocks.back();
      free_blocks.pop_back();
      TORCH_INTERNAL_ASSERT(!block->allocated_);
      block->allocated_ = true;
      return block;
    }
    return nullptr;
  }

  void CUDAStreamCapturableCachingHostAllocator::free_finished_allocations_under_lock(PrivatePool& cuda_mem_pool) {
    // m_ must be held.

    // WARNING: This is O(N), where N is number of live blocks.
    for (auto block_it = cuda_mem_pool.blocks.begin(); block_it != cuda_mem_pool.blocks.end();) {
      GraphBlock *block = *block_it;
      if (block->allocated_) {
        block_it++;
        continue;
      }
      // is it okay to insert into empty nodes into any random stream?
      // This a mutation, so it's not great... Though it *shouldn't*
      // have any user-perceptible side effects.
      CUDAStream stream = getCurrentCUDAStream();
      cudaGraphNode_t destination_node = insert_empty_node_under_lock(nullptr, stream);

      bool all_usages_done = true;

      // there is a problem with this approach! What if we capture to
      // two graphs with the same memory pool? We have no way to detect
      // that "output allocations" of a graph are done being used...

      // Actually, in this case, there won't be a path from the using
      // nodes to the destination nodes, since the two graphs are
      // disjoint. So the output allocation will never be recycled,
      // which is "safe".
      for (cudaGraphNode_t source_node: block->using_nodes) {
        if (!path_exists(source_node, destination_node)) {
          all_usages_done = false;
        }
      }

      if (all_usages_done) {
        block->using_nodes.clear();
        // should be using a mutex here lol
        auto index = size_index(block->size);
        std::deque<GraphBlock*>& free_blocks = cuda_mem_pool.free_list_[index];
        free_blocks.push_front(block);
        block_it = cuda_mem_pool.blocks.erase(block_it);
      } else {
        block_it++;
      }
    }
  }


at::cuda::CUDAStreamCapturableCachingHostAllocator* getCUDAStreamCapturableCachingHostAllocator() {
  return &stream_capturable_caching_host_allocator;
}


namespace {
  static at::HostAllocatorRegistry
  g_host_allocator_registry_instance(at::kCUDA, []() -> at::HostAllocator* {
    cudaStreamCaptureStatus capture_status{cudaStreamCaptureStatusNone};
    AT_CUDA_CHECK(cudaStreamIsCapturing(getCurrentCUDAStream(), &capture_status));
    if (capture_status == cudaStreamCaptureStatusNone) {
      return &caching_host_allocator;
    } else {
      return &stream_capturable_caching_host_allocator;
    }
  });
}
} // namespace at::cuda
