#include "ReIDResourceManager.h"
#include <cstring>
#include <iostream>

namespace reid {

ReIDResourceManager::ReIDResourceManager()
    : h_base_ptr_(nullptr)
    , d_base_ptr_(nullptr)
    , pool_size_(0)
    , initialized_(false)
    , total_submitted_(0)
    , total_completed_(0)
    , total_dropped_(0) {
}

ReIDResourceManager::~ReIDResourceManager() {
    release();
}

bool ReIDResourceManager::initialize(uint32_t pool_size) {
    if (initialized_) {
        return true;
    }

    pool_size_ = pool_size;

    if (!allocateMemoryPool(pool_size)) {
        return false;
    }

    tensor_pool_.resize(pool_size);

    for (uint32_t i = 0; i < pool_size; ++i) {
        TensorSlot& slot = tensor_pool_[i];
        slot.slot_id = i;
        slot.state.store(SlotState::FREE);

        slot.h_embedding_ptr = reinterpret_cast<float*>(
            reinterpret_cast<uint8_t*>(h_base_ptr_) + (i * 1024)
        );
        slot.d_output_ptr = reinterpret_cast<uint8_t*>(d_base_ptr_) + (i * 1024);

        slot.d_input_ptr = nullptr;
        slot.input_tensor = nullptr;
        slot.output_tensor = nullptr;
        slot.preprocessing_done_event = nullptr;
        slot.inference_done_event = nullptr;

        free_queue_.push(i);
    }

    initialized_ = true;
    return true;
}

void ReIDResourceManager::release() {
    if (!initialized_) {
        return;
    }

    free_queue_.clear();
    pending_queue_.clear();
    completion_queue_.clear();

    freeMemoryPool();

    tensor_pool_.clear();
    initialized_ = false;
}

bool ReIDResourceManager::allocateMemoryPool(uint32_t pool_size) {
    size_t total_size = pool_size * 1024;

    std::memset(&h_base_ptr_, 0, sizeof(h_base_ptr_));
    std::memset(&d_base_ptr_, 0, sizeof(d_base_ptr_));

    h_base_ptr_ = malloc(total_size);
    if (!h_base_ptr_) {
        std::cerr << "Failed to allocate embedding pool memory" << std::endl;
        return false;
    }

    d_base_ptr_ = h_base_ptr_;

    std::memset(h_base_ptr_, 0, total_size);

    std::cout << "Allocated " << total_size / 1024 << " KB for embedding pool ("
              << pool_size << " slots)" << std::endl;

    return true;
}

void ReIDResourceManager::freeMemoryPool() {
    if (h_base_ptr_) {
        free(h_base_ptr_);
        h_base_ptr_ = nullptr;
        d_base_ptr_ = nullptr;
    }
}

bool ReIDResourceManager::acquireFreeSlot(uint32_t& out_slot_id) {
    if (free_queue_.try_pop(out_slot_id)) {
        total_submitted_.fetch_add(1);
        tensor_pool_[out_slot_id].state.store(SlotState::PREPROCESSING);
        return true;
    }
    total_dropped_.fetch_add(1);
    return false;
}

void ReIDResourceManager::pushPendingTask(uint32_t slot_id) {
    tensor_pool_[slot_id].state.store(SlotState::PENDING);
    pending_queue_.push(slot_id);
}

bool ReIDResourceManager::popPendingTask(uint32_t& out_slot_id, int timeout_ms) {
    if (pending_queue_.wait_pop(out_slot_id, timeout_ms)) {
        tensor_pool_[out_slot_id].state.store(SlotState::INFERRING);
        return true;
    }
    return false;
}

void ReIDResourceManager::pushCompletion(uint32_t slot_id) {
    tensor_pool_[slot_id].state.store(SlotState::COMPLETED);
    completion_queue_.push(slot_id);
    total_completed_.fetch_add(1);
}

bool ReIDResourceManager::popCompletion(uint32_t& out_slot_id, int timeout_ms) {
    return completion_queue_.wait_pop(out_slot_id, timeout_ms);
}

void ReIDResourceManager::recycleSlot(uint32_t slot_id) {
    tensor_pool_[slot_id].state.store(SlotState::FREE);
    free_queue_.push(slot_id);
}

TensorSlot* ReIDResourceManager::getTensorSlot(uint32_t slot_id) {
    if (slot_id >= pool_size_) {
        return nullptr;
    }
    return &tensor_pool_[slot_id];
}

ReIDPreparedInput* ReIDResourceManager::getPreparedInput(uint32_t slot_id) {
    if (slot_id >= pool_size_) {
        return nullptr;
    }
    return &tensor_pool_[slot_id].prepared_input;
}

float* ReIDResourceManager::getOutputEmbeddingBuffer(uint32_t slot_id) {
    if (slot_id >= pool_size_) {
        return nullptr;
    }
    return tensor_pool_[slot_id].h_embedding_ptr;
}

void* ReIDResourceManager::getDeviceOutputPtr(uint32_t slot_id) {
    if (slot_id >= pool_size_) {
        return nullptr;
    }
    return tensor_pool_[slot_id].d_output_ptr;
}

void* ReIDResourceManager::getDeviceInputPtr(uint32_t slot_id) {
    if (slot_id >= pool_size_) {
        return nullptr;
    }
    return tensor_pool_[slot_id].d_input_ptr;
}

void* ReIDResourceManager::getPreprocessingDoneEvent(uint32_t slot_id) {
    if (slot_id >= pool_size_) {
        return nullptr;
    }
    return tensor_pool_[slot_id].preprocessing_done_event;
}

void* ReIDResourceManager::getInferenceDoneEvent(uint32_t slot_id) {
    if (slot_id >= pool_size_) {
        return nullptr;
    }
    return tensor_pool_[slot_id].inference_done_event;
}

PipelineStats ReIDResourceManager::getStats() const {
    PipelineStats stats;
    stats.free_slots = static_cast<uint32_t>(free_queue_.size());
    stats.pending_tasks = static_cast<uint32_t>(pending_queue_.size());
    stats.completed_tasks = static_cast<uint32_t>(completion_queue_.size());
    stats.total_submitted = total_submitted_.load();
    stats.total_completed = total_completed_.load();
    stats.total_dropped = total_dropped_.load();
    stats.dla0_utilization = 0.0f;
    stats.dla1_utilization = 0.0f;
    return stats;
}

}
