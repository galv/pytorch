#pragma once

#include <ATen/core/CachingHostAllocator.h>
#include <c10/core/Allocator.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Deprecated.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>

namespace at::cuda {

template <
    typename S,
    typename E,
    typename B,
    typename Child>
struct CachingHostAllocatorImpl: std::enable_shared_from_this<Child> {
 private:
    Child& self() {
        return *static_cast<Child*>(this);
    }
    const Child& self() const {
        return *static_cast<const Child*>(this);
    }

 public:
  explicit CachingHostAllocatorImpl() {}

  virtual ~CachingHostAllocatorImpl() {
    active_ = false;
    if (pinned_use_background_threads()) {
      getBackgroundThreadPool()->waitWorkComplete();
    }
  }

 public:
  // return data_ptr and block pair.
  virtual std::pair<void*, void*> allocate(size_t size) {
    if (size == 0) {
      return {nullptr, nullptr};
    }

    // If we are using background threads, we can process events in the
    // background.
    if (!pinned_use_background_threads()) {
      process_events();
    }

    // Round up the allocation to the nearest power of two to improve reuse.
    // These power of two sizes are also used to index into the free list.
    size_t roundSize = c10::llvm::PowerOf2Ceil(size);

    // First, try to allocate from the free list
    auto* block = get_free_block(roundSize);
    if (block) {
      return {block->ptr_, reinterpret_cast<void*>(block)};
    }

    // Check in the recently freed blocks with pending events to see if we
    // can reuse them. Call get_free_block again after processing events
    if (pinned_use_background_threads()) {
      process_events_for_specific_size(roundSize);
      block = get_free_block(roundSize);
      if (block) {
        return {block->ptr_, reinterpret_cast<void*>(block)};
      }

      // Launch the background thread and process events in a loop.
      static bool background_thread_flag [[maybe_unused]] = [this] {
        getBackgroundThreadPool()->run([&]() {
          while (active_) {
            process_events();
            std::this_thread::sleep_for(std::chrono::microseconds(100));
          }
        });
        return true;
      }();
    }

    // Slow path: if we can't allocate from the cached free list, we need
    // to create a new block.
    void* ptr = nullptr;
    allocate_host_memory(roundSize, &ptr);

    // Then, create a new block.
    block = new B(roundSize, ptr, self().shared_from_this());
    block->allocated_ = true;

    add_allocated_block(block);
    return {block->ptr_, reinterpret_cast<void*>(block)};
  }

  virtual void free(void* ctx) {
    if (!ctx) {
      return;
    }

    // Note: we can assume that free is correctly paired with alloc, and thus we
    // do not need to look up the ctx in blocks_.
    auto* block = reinterpret_cast<B*>(ctx);

    std::optional<std::vector<E>> events;
    {
      std::lock_guard<std::mutex> g(block->mutex_);
      block->allocated_ = false;
      if (block->streams_.empty()) {
        TORCH_INTERNAL_ASSERT(block->event_count_ == 0);
      } else {
        events = std::vector<E>();
        events->reserve(block->streams_.size());
        for (auto stream : block->streams_) {
          record_stream(events, stream);
        }
        block->event_count_ += events->size();
        block->streams_.clear();
      }
    }

    if (!events) {
      auto index = size_index(block->size_);
      std::lock_guard<std::mutex> g(free_list_[index].mutex_);
      free_list_[index].list_.push_back(block);
      stats_.allocation_bucket_stats[index].decrease(1);
      stats_.allocated_bytes_bucket_stats[index].decrease(block->size_);
    } else {
      // restore these events that record by used streams.
      std::lock_guard<std::mutex> g(events_mutex_);
      for (auto&& event : *events) {
        events_.emplace_front(std::move(event), block);
      }
    }
  }

  virtual bool record_event(void* ptr, void* ctx, c10::Stream s) {
    S stream = S(s);
    auto* block = reinterpret_cast<B*>(ctx);

    // Note: we need to check if the passed-in `ctx` is valid. This is because
    // `record_event` (via `CachingHostAllocator_recordEvent`) can be invoked on
    // an arbitrary tensor, and is not guaranteed to correspond to a pinned
    // memory allocation. Therefore, we need to check that `ctx` is valid before
    // proceeding.
    {
      std::lock_guard<std::mutex> g(blocks_mutex_);
      if (blocks_.find(block) != blocks_.end()) {
        // Now we know this object is safe to access.
        std::lock_guard<std::mutex> gb(block->mutex_);
        TORCH_INTERNAL_ASSERT(block->allocated_);
        block->streams_.insert(stream);
        return true;
      }
      auto it = ptr_to_block_.find(ptr);
      if (it != ptr_to_block_.end()) {
        block = it->second;
        std::lock_guard<std::mutex> g(block->mutex_);
        TORCH_INTERNAL_ASSERT(block->allocated_);
        block->streams_.insert(stream);
        return true;
      }
    }

    return false;
  }

  virtual void empty_cache() {
    // Flush any available blocks into the free_list.
    process_events();

    // Remove all elements from the free list, remove them from the blocks
    // list, and free the associated pinned memory allocation. This requires
    // concurrently holding both the free list mutexes and the blocks mutex, and
    // is the only function that concurrently holds multiple mutexes.
    for (size_t i = 0; i < free_list_.size(); ++i) {
      std::lock(free_list_[i].mutex_, blocks_mutex_);
      std::lock_guard<std::mutex> gf(free_list_[i].mutex_, std::adopt_lock);
      std::lock_guard<std::mutex> gb(blocks_mutex_, std::adopt_lock);

      std::vector<B*> blocks_to_remove(free_list_[i].list_.begin(), free_list_[i].list_.end());
      free_list_[i].list_.clear();

      for (auto* block : blocks_to_remove) {
        blocks_.erase(block);
        ptr_to_block_.erase(block->ptr_);
        stats_.allocation.decrease(1);
        stats_.allocated_bytes.decrease(block->size_);
        free_block(block);
        delete block;
      }
    }
  }

  inline size_t size_index(size_t size) {
    return c10::llvm::Log2_64_Ceil(size);
  }

  virtual bool pinned_use_background_threads() {
    return false;
  }

  virtual void copy_data(void* dest [[maybe_unused]], const void* src [[maybe_unused]], std::size_t count [[maybe_unused]]) const {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "Not implemented for copy_data");
  }

  HostStats getStats() {
    HostStats stats;

    // To keep getStats lightweight we do *not* flush any available blocks
    // into the free_list. This may skew the stats a bit.

    auto add_bucket_stats = [](Stat& accumulator, const Stat& other) {
      accumulator.allocated += other.allocated;
      accumulator.current += other.current;
      accumulator.freed += other.freed;
      // Since peaks are measured per bucket independently, we add them up
      // to estimate the total peak. This is not strictly correct, but it is
      // the best approximation we can get after the fact.
      accumulator.peak += other.peak;
    };

    // Accurate reading of memory stats requires concurrently holding both the
    // free list mutexes and the blocks mutex. Previously, this was only done in
    // empty_cache function.
    for (size_t i = 0; i < free_list_.size(); ++i) {
      std::lock(free_list_[i].mutex_, blocks_mutex_);
      std::lock_guard<std::mutex> gf(free_list_[i].mutex_, std::adopt_lock);
      std::lock_guard<std::mutex> gb(blocks_mutex_, std::adopt_lock);

      // We collect the slow-path stats only once, since they are not collected
      // per bucket (we pick index 0 arbitrarily). These are also all the host
      // allocations, not taking into account caching and free lists.
      if (i == 0) {
        stats.segment = stats_.allocation;
        stats.reserved_bytes = stats_.allocated_bytes;
        stats.num_host_alloc = stats.segment.allocated;
        stats.num_host_free = stats.segment.freed;
      }

      // Bucket stats need to be merged with the slow-path stats. We do this in
      // a best effort manner, since we can't really replay the cached events per bucket.
      add_bucket_stats(stats.allocation, stats_.allocation_bucket_stats[i]);
      add_bucket_stats(stats.allocated_bytes, stats_.allocated_bytes_bucket_stats[i]);
    }

    // Get the timing stats
    {
      std::lock_guard<std::mutex> g(stats_.timing_mutex_);

      stats.host_alloc_time = stats_.host_alloc_time;
      stats.host_free_time = stats_.host_free_time;
    }

    return stats;
  }

  void resetAccumulatedStats() {
    // Reseting accumulated memory stats requires concurrently holding both the
    // free list mutexes and the blocks mutex. Previously, this was only done in
    // empty_cache function.
    for (size_t i = 0; i < free_list_.size(); ++i) {
      std::lock(free_list_[i].mutex_, blocks_mutex_);
      std::lock_guard<std::mutex> gf(free_list_[i].mutex_, std::adopt_lock);
      std::lock_guard<std::mutex> gb(blocks_mutex_, std::adopt_lock);

      if (i == 0) {
        stats_.allocation.reset_accumulated();
        stats_.allocated_bytes.reset_accumulated();
      }
      stats_.allocation_bucket_stats[i].reset_accumulated();
      stats_.allocated_bytes_bucket_stats[i].reset_accumulated();
    }

    // Also reset timing stats
    {
      std::lock_guard<std::mutex> g(stats_.timing_mutex_);
      stats_.host_alloc_time.reset_accumulated();
      stats_.host_free_time.reset_accumulated();
    }
  }

  void resetPeakStats() {
    // Reseting peak memory stats requires concurrently holding both the
    // free list mutexes and the blocks mutex. Previously, this was only done in
    // empty_cache function.
    for (size_t i = 0; i < free_list_.size(); ++i) {
      std::lock(free_list_[i].mutex_, blocks_mutex_);
      std::lock_guard<std::mutex> gf(free_list_[i].mutex_, std::adopt_lock);
      std::lock_guard<std::mutex> gb(blocks_mutex_, std::adopt_lock);

      if (i == 0) {
        stats_.allocation.reset_peak();
        stats_.allocated_bytes.reset_peak();
      }
      stats_.allocation_bucket_stats[i].reset_peak();
      stats_.allocated_bytes_bucket_stats[i].reset_peak();
    }

    // Also reset timing stats
    {
      std::lock_guard<std::mutex> g(stats_.timing_mutex_);
      stats_.host_alloc_time.reset_peak();
      stats_.host_free_time.reset_peak();
    }
  }

 private:
  virtual void add_allocated_block(B* block) {
    std::lock_guard<std::mutex> g(blocks_mutex_);
    blocks_.insert(block);
    stats_.allocation.increase(1);
    stats_.allocated_bytes.increase(block->size_);
    ptr_to_block_.insert({block->ptr_, block});

    // Unfortunately, we have to, on the slow path, quickly
    // lock the bucket to record the allocation. This should
    // be a rare event once the cache is warmed up.
    auto size = block->size_;
    auto index = size_index(size);
    {
      std::lock_guard<std::mutex> g(free_list_[index].mutex_);
      stats_.allocation_bucket_stats[index].increase(1);
      stats_.allocated_bytes_bucket_stats[index].increase(size);
    }
  }

  virtual B* get_free_block(size_t size) {
    auto index = size_index(size);
    std::lock_guard<std::mutex> g(free_list_[index].mutex_);
    if (!free_list_[index].list_.empty()) {
      B* block = free_list_[index].list_.back();
      free_list_[index].list_.pop_back();
      block->allocated_ = true;
      stats_.allocation_bucket_stats[index].increase(1);
      stats_.allocated_bytes_bucket_stats[index].increase(size);
      return block;
    }
    return nullptr;
  }

  virtual void process_events() {
    // process all events until the last unready event, not for specific size.
    process_events_for_specific_size(-1);
  }

  // If size is -1, process all events from backwards until the last unready
  // event. Otherwise, process events for a specific size and on first ready block
  // is found, add it to the free list and return.
  virtual void process_events_for_specific_size(int64_t size) {
    size_t event_count = 0;
    size_t max_events = 0;
    {
      std::lock_guard<std::mutex> g(events_mutex_);
      max_events = events_.size();
    }

    while (true) {
      // Avoid calling cudaEventDestroy while holding a mutex, so move
      // intermediate events out of the lock into this object.
      // process the last event
      std::optional<std::pair<E, B*>> processed;
      {
        std::lock_guard<std::mutex> g(events_mutex_);
        if (!events_.empty()) {
          processed = std::move(events_.back());
          events_.pop_back();
        }
      }

      if (!processed) {
        return;
      }

      if (size != -1) {
        if (event_count++ > max_events) {
          {
            std::lock_guard<std::mutex> g(events_mutex_);
            events_.push_front(std::move(*processed));
          }
          return;
        }
        if (size != (int64_t)processed->second->size_) {
          // if we are processing a specific size, and the size of the block
          // doesn't match, we can't use it.
          {
            std::lock_guard<std::mutex> g(events_mutex_);
            events_.push_front(std::move(*processed));
          }
          continue;
        }
      }

      // otherwise, query the event
      {
        // now, see if we can handle this element
        auto& event = processed->first;
        if (!query_event(event)) {
          // push the event onto the back if it's not ready.
          {
            std::lock_guard<std::mutex> g(events_mutex_);
            if (size == -1) {
              events_.push_back(std::move(*processed));
              return;
            } else {
              events_.push_front(std::move(*processed));
              continue;
            }
          }
        }
      }

      // Process the events.
      TORCH_INTERNAL_ASSERT(processed);
      auto* block = processed->second;
      bool available = false;
      {
        std::lock_guard<std::mutex> g(block->mutex_);
        TORCH_INTERNAL_ASSERT(!block->allocated_)
        block->event_count_--;
        if (block->event_count_ == 0) {
          available = true;
        }
      }

      if (available) {
        auto index = size_index(block->size_);
        std::lock_guard<std::mutex> g(free_list_[index].mutex_);
        free_list_[index].list_.push_back(block);
        stats_.allocation_bucket_stats[index].decrease(1);
        stats_.allocated_bytes_bucket_stats[index].decrease(size);
        if (size != -1) {
          return;
        }
      }
    }
  }

  TaskThreadPool* getBackgroundThreadPool() {
    static TaskThreadPool* pool = new TaskThreadPool(1);
    return pool;
  }

  /* These following functions are runtime-related. */

  // Allocate page-locked memory on the host.
  virtual void allocate_host_memory(size_t size, void** ptr) {
    TORCH_CHECK_NOT_IMPLEMENTED(
        false, "Not implemented for allocate_host_memory");
  }

  // Free block and release the pointer contained in block.
  virtual void free_block(B* block) {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "Not implemented for free_block");
  }

  // Record an event on stream and store event into events.
  virtual void record_stream(std::optional<std::vector<E>>& events, S stream) {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "Not implemented for record_stream");
  }

  // Query event if it is completed.
  virtual bool query_event(E& event) {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "Not implemented for query_event");
  }

  alignas(64) std::mutex blocks_mutex_;
  ska::flat_hash_set<B*> blocks_; // block list
  ska::flat_hash_map<void*, B*> ptr_to_block_;

  // We keep free list as a vector of free lists, one for each power of two
  // size. This allows us to quickly find a free block of the right size.
  // We use deque to store per size free list and guard the list with its own
  // mutex.
  alignas(64) std::vector<FreeBlockList<B>> free_list_ =
      std::vector<FreeBlockList<B>>(MAX_SIZE_INDEX);

  alignas(64) std::mutex events_mutex_;
  std::deque<std::pair<E, B*>> events_; // event queue paired with block

  // Indicates whether the object is active.
  // Set to false in the destructor to signal background threads to stop.
  std::atomic<bool> active_{true};
protected:
  alignas(64) HostStatsStaged stats_;
};

template <typename T, c10::DeleterFnPtr deleteFunc>
struct CachingHostAllocatorInterface : public HostAllocator {
  CachingHostAllocatorInterface() : impl_(std::make_shared<T>()) {}

  using Block = typename T::Block;

  at::DataPtr allocate(size_t size) override {
    T* impl = getImplByFilter(getCurrentCUDAStream());
    auto ptr_and_ctx = impl->allocate(size);
    return {
        ptr_and_ctx.first,
        ptr_and_ctx.second,
        deleteFunc, // Use the template parameter deleter function
        at::DeviceType::CPU};
  }

  void free(void* ctx) {
    TORCH_INTERNAL_ASSERT(ctx);
    auto* block = reinterpret_cast<Block*>(ctx);
    block->pool_ptr_->free(ctx);
  }

  bool record_event(void* ptr, void* ctx, c10::Stream stream) override {
    // Warning: Sometimes a non-block ctx is passed to record event
    // Figure out what to do then.
    T* impl = getImplByFilter(getCurrentCUDAStream());
    return impl->record_event(ptr, ctx, stream);
  }

  void begin_allocate_to_pool(
      c10::MempoolId_t pool_id,
      std::function<bool(c10::Stream)> filter) override {
    for (auto it = captures_underway_.begin(); it != captures_underway_.end();
         ++it) {
      TORCH_CHECK(
          it->first != pool_id,
          "beginAllocateToPool: already recording to pool_id");
    }
    // We are doing emplace back, but iterating over
    // captures_underway_ from front to back. This prevents us from
    // having a LIFO-based way of doing allocation to custom pools
    // when nesting torch.cuda.memory.use_mem_pool context managers.
    // Does the CUDACachingAllocator have the same problem?
    captures_underway_.emplace_back(pool_id, std::move(filter));
    getOrCreateImpl(pool_id);
  }

  // TODO: Use a mutex to guard this object. Maybe a reader writer lock.

  void end_allocate_to_pool(c10::MempoolId_t pool_id) override {
    auto it = std::find_if(captures_underway_.begin(), captures_underway_.end(),
        [pool_id](const auto& elem) {
            return std::get<0>(elem) == pool_id;
        });
    TORCH_INTERNAL_ASSERT(it != captures_underway_.end());
    captures_underway_.erase(it);
  }

  void release_pool(c10::MempoolId_t pool_id) override {
    auto it = private_impls_.find(pool_id);
    TORCH_INTERNAL_ASSERT(it != private_impls_.end());
    // TODO: Figure out if this should be called or not. I have no
    // idea, but it seems like a good idea. I think it's necessary?
    // But what if I have an allocation that stays live after the
    // memory pool is finished? I still need to keep the impl alive,
    // don't I? Ugh... std::shared_ptr<> in my current design does successfully do that fortunately.
    it->second.front()->empty_cache();
    if (it->second.empty()) {
      private_impls_.erase(it);
    }
  }

  // TODO: I think I need a move constructor that converts a HostBlock
  // here to a StreamCapturableHostBlock. That would work.

  void empty_cache(/*c10::MempoolId_t mempool_id = {0, 0}*/) override {
    // if (mempool_id.first == 0 && mempool_id.second == 0 &&
    //     captures_underway_.empty()) {
      
    // }
    // // TODO: How do we guarantee that empty_cache never gets called on
    // // a pool used for stream capture? I'm not sure...
    // T* impl = getImpl(mempool_id);
    // impl->empty_cache();
    impl_->empty_cache();
  }

  void copy_data(void* dest, const void* src, std::size_t count)
      const override {
    impl_->copy_data(dest, src, count);
  }

  HostStats get_stats() override {
    return impl_->getStats();
  }

  void reset_accumulated_stats() override {
    impl_->resetAccumulatedStats();
  }

  void reset_peak_stats() override {
    impl_->resetPeakStats();
  }

  T* getImplByFilter(c10::cuda::CUDAStream stream) {
    // iterate in reverse order to enforce a LIFO ordering when two
    // nested memory pools have filters that return true. See
    // https://github.com/pytorch/pytorch/issues/161193
    for (auto it = captures_underway_.rbegin(); it != captures_underway_.rend(); ++it) {
      auto &&[mempool_id, filter] = *it;
      if (filter(stream)) {
        return private_impls_.at(mempool_id).front().get();
      }
    }
    return impl_.get();
  }

  T* getImpl(c10::MempoolId_t pool_id) {
    if (pool_id == c10::MempoolId_t{0, 0}) {
      return impl_.get();
    }
    auto it = private_impls_.find(pool_id);
    TORCH_INTERNAL_ASSERT(it != private_impls_.end());
    return it->second.front().get();
  }

  void getOrCreateImpl(c10::MempoolId_t pool_id) {
    TORCH_INTERNAL_ASSERT(pool_id != (c10::MempoolId_t{0, 0}));
    auto it = private_impls_.find(pool_id);
    if (it == private_impls_.end()) {
      private_impls_[pool_id].emplace_back(std::make_shared<T>());
    } else {
      private_impls_.at(pool_id).emplace_back(private_impls_.at(pool_id).front());
    }
  }

  ska::flat_hash_map<
      c10::MempoolId_t,
      std::vector<std::shared_ptr<T>>,
      c10::MempoolIdHash>
      private_impls_;

  std::vector<std::pair<MempoolId_t, std::function<bool(c10::Stream)>>> captures_underway_;

  std::shared_ptr<T> impl_;
};

struct GraphBlock;
struct PrivatePool;

// Struct containing memory allocator summary statistics for host, as they
// are staged for reporting. This is a temporary struct that is used to
// avoid locking the allocator while collecting stats.
struct HostStatsStagedStreamCapturable {
  // COUNT: allocations requested by client code resulting in a new segment/block allocation
  Stat allocation;
  // SUM: bytes within active memory blocks, including blocks that are
  // currently in the free list.
  Stat allocated_bytes;
  // COUNT: number of allocations per bucket
  std::vector<Stat> allocation_bucket_stats = std::vector<Stat>(MAX_SIZE_INDEX);
  // SUM: bytes of allocation per bucket
  std::vector<Stat> allocated_bytes_bucket_stats = std::vector<Stat>(MAX_SIZE_INDEX);
};

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
