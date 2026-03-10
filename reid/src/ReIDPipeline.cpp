#include "ReIDPipeline.h"
#include <iostream>
#include <cstring>

namespace reid {

ReIDPipeline::ReIDPipeline()
    : ctx_(nullptr)
    , initialized_(false) {
}

ReIDPipeline::~ReIDPipeline() {
    shutdown();
}

bool ReIDPipeline::initialize(dwContextHandle_t ctx, const ReIDPipelineConfig& config) {
    if (initialized_) {
        return true;
    }

    if (!ctx) {
        std::cerr << "ReIDPipeline: Invalid context" << std::endl;
        return false;
    }

    ctx_ = ctx;

    if (!resource_manager_.initialize(config.pool_size)) {
        std::cerr << "ReIDPipeline: Failed to initialize resource manager" << std::endl;
        return false;
    }

    if (!preprocessor_.initialize(ctx, &resource_manager_)) {
        std::cerr << "ReIDPipeline: Failed to initialize preprocessor" << std::endl;
        resource_manager_.release();
        return false;
    }

    if (!worker_pool_.initialize(ctx, config.engine_path_dla0, config.engine_path_dla1,
                                 &resource_manager_)) {
        std::cerr << "ReIDPipeline: Failed to initialize worker pool" << std::endl;
        preprocessor_.release();
        resource_manager_.release();
        return false;
    }

    if (config.enable_global_association) {
        if (!global_association_.initialize(&resource_manager_)) {
            std::cerr << "ReIDPipeline: Failed to initialize global association" << std::endl;
            worker_pool_.release();
            preprocessor_.release();
            resource_manager_.release();
            return false;
        }
        global_association_.start();
    }

    worker_pool_.start();

    initialized_ = true;

    std::cout << "ReIDPipeline: Initialized successfully with pool size "
              << config.pool_size << std::endl;

    return true;
}

void ReIDPipeline::shutdown() {
    if (!initialized_) {
        return;
    }

    std::cout << "ReIDPipeline: Shutting down..." << std::endl;

    worker_pool_.stop();
    global_association_.stop();

    worker_pool_.release();
    global_association_.release();
    preprocessor_.release();
    resource_manager_.release();

    initialized_ = false;

    std::cout << "ReIDPipeline: Shutdown complete" << std::endl;
}

bool ReIDPipeline::submitROI(const ReIDRequest& request) {
    if (!initialized_) {
        return false;
    }

    uint32_t slot_id;
    if (!resource_manager_.acquireFreeSlot(slot_id)) {
        return false;
    }

    if (!preprocessor_.prepareAsync(request, slot_id)) {
        resource_manager_.recycleSlot(slot_id);
        return false;
    }

    return true;
}

size_t ReIDPipeline::pollResults(ReIDResult* results, size_t maxResults) {
    if (!initialized_ || !results || maxResults == 0) {
        return 0;
    }

    size_t count = 0;
    for (size_t i = 0; i < maxResults; ++i) {
        uint32_t slot_id;
        if (!resource_manager_.popCompletion(slot_id, 0)) {
            break;
        }

        ReIDPreparedInput* input = resource_manager_.getPreparedInput(slot_id);
        float* embedding = resource_manager_.getOutputEmbeddingBuffer(slot_id);

        results[count].local_track_id = input->local_track_id;
        results[count].camera_id = input->camera_id;
        results[count].timestamp_us = input->timestamp_us;
        results[count].source_bbox = input->source_bbox;
        std::memcpy(results[count].embedding, embedding, REID_FEAT_DIM * sizeof(float));
        results[count].global_track_id = 0;
        results[count].association_confidence = 0.0f;

        resource_manager_.recycleSlot(slot_id);

        count++;
    }

    return count;
}

PipelineStats ReIDPipeline::getStats() const {
    if (!initialized_) {
        PipelineStats stats = {};
        return stats;
    }

    return resource_manager_.getStats();
}

}
