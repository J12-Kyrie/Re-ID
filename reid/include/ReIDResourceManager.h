#ifndef REID_RESOURCE_MANAGER_H
#define REID_RESOURCE_MANAGER_H

#include "ReIDTypes.h"
#include "ThreadSafeQueue.h"
#include <vector>
#include <atomic>

namespace reid {

class ReIDResourceManager {
public:
    ReIDResourceManager();
    ~ReIDResourceManager();

    ReIDResourceManager(const ReIDResourceManager&) = delete;
    ReIDResourceManager& operator=(const ReIDResourceManager&) = delete;

    bool initialize(uint32_t pool_size);
    void release();

    // Slot validation for YOLO-managed slots
    bool isValidSlotId(uint32_t slot_id) const;

    // Simplified slot management - no acquisition needed since YOLO manages slots
    void pushPendingTask(uint32_t slot_id);
    bool popPendingTask(uint32_t& out_slot_id, int timeout_ms = -1);
    void pushCompletion(uint32_t slot_id);
    bool popCompletion(uint32_t& out_slot_id, int timeout_ms = 0);
    void recycleSlot(uint32_t slot_id);

    TensorSlot* getTensorSlot(uint32_t slot_id);
    ReIDPreparedInput* getPreparedInput(uint32_t slot_id);
    float* getOutputEmbeddingBuffer(uint32_t slot_id);
    void* getDeviceOutputPtr(uint32_t slot_id);
    void* getDeviceInputPtr(uint32_t slot_id);
    void* getPreprocessingDoneEvent(uint32_t slot_id);
    void* getInferenceDoneEvent(uint32_t slot_id);

    PipelineStats getStats() const;

    bool isInitialized() const { return initialized_; }

private:
    bool allocateMemoryPool(uint32_t pool_size);
    void freeMemoryPool();

    std::vector<TensorSlot> tensor_pool_;

    ThreadSafeQueue<uint32_t> free_queue_;
    ThreadSafeQueue<uint32_t> pending_queue_;
    ThreadSafeQueue<uint32_t> completion_queue_;

    void* h_base_ptr_;
    void* d_base_ptr_;

    uint32_t pool_size_;
    bool initialized_;

    std::atomic<uint64_t> total_submitted_;
    std::atomic<uint64_t> total_completed_;
    std::atomic<uint64_t> total_dropped_;
};

}

#endif
