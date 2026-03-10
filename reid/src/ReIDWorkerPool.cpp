#include "ReIDWorkerPool.h"
#include <iostream>

namespace reid {

ReIDWorkerPool::ReIDWorkerPool()
    : pool_manager_(nullptr)
    , running_(false)
    , initialized_(false) {
}

ReIDWorkerPool::~ReIDWorkerPool() {
    release();
}

bool ReIDWorkerPool::initialize(dwContextHandle_t ctx,
                                const char* engine_path_dla0,
                                const char* engine_path_dla1,
                                ReIDResourceManager* pool_manager) {
    if (initialized_) {
        return true;
    }

    if (!pool_manager) {
        std::cerr << "ReIDWorkerPool: Invalid pool manager" << std::endl;
        return false;
    }

    pool_manager_ = pool_manager;

    ReIdWorkerConfig config0;
    config0.enginePath = engine_path_dla0;
    config0.dlaCore = 0;
    config0.conditionerConfig.engineHasPreprocess = false;

    if (!worker0_.init(ctx, &config0)) {
        std::cerr << "ReIDWorkerPool: Failed to initialize worker 0" << std::endl;
        return false;
    }

    ReIdWorkerConfig config1;
    config1.enginePath = engine_path_dla1;
    config1.dlaCore = 1;
    config1.conditionerConfig.engineHasPreprocess = false;

    if (!worker1_.init(ctx, &config1)) {
        std::cerr << "ReIDWorkerPool: Failed to initialize worker 1" << std::endl;
        worker0_.release();
        return false;
    }

    initialized_ = true;
    return true;
}

void ReIDWorkerPool::release() {
    if (!initialized_) {
        return;
    }

    stop();

    worker0_.release();
    worker1_.release();

    initialized_ = false;
}

void ReIDWorkerPool::start() {
    if (!initialized_ || running_.load()) {
        return;
    }

    running_.store(true);

    worker_thread0_ = std::thread(&ReIDWorkerPool::workerThreadLoop, this, 0);
    worker_thread1_ = std::thread(&ReIDWorkerPool::workerThreadLoop, this, 1);

    std::cout << "ReIDWorkerPool: Started dual-DLA workers" << std::endl;
}

void ReIDWorkerPool::stop() {
    if (!running_.load()) {
        return;
    }

    running_.store(false);

    if (worker_thread0_.joinable()) {
        worker_thread0_.join();
    }
    if (worker_thread1_.joinable()) {
        worker_thread1_.join();
    }

    std::cout << "ReIDWorkerPool: Stopped workers" << std::endl;
}

void ReIDWorkerPool::workerThreadLoop(int dla_core_id) {
    ReIdWorker* worker = (dla_core_id == 0) ? &worker0_ : &worker1_;

    while (running_.load()) {
        uint32_t slot_id;
        if (!pool_manager_->popPendingTask(slot_id, 100)) {
            continue;
        }

        TensorSlot* slot = pool_manager_->getTensorSlot(slot_id);
        if (!slot) {
            continue;
        }

        ReIDPreparedInput* input = &slot->prepared_input;
        std::vector<float> embedding;

        bool success = worker->process(nullptr, input->source_bbox, &embedding);

        if (success && embedding.size() == REID_FEAT_DIM) {
            float* output = pool_manager_->getOutputEmbeddingBuffer(slot_id);
            std::memcpy(output, embedding.data(), REID_FEAT_DIM * sizeof(float));
        } else {
            std::cerr << "ReIDWorkerPool: DLA " << dla_core_id
                     << " inference failed for slot " << slot_id << std::endl;
        }

        pool_manager_->pushCompletion(slot_id);
    }
}

}
