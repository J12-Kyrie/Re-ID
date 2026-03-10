#include "ReIDPreprocessor.h"
#include <iostream>

namespace reid {

ReIDPreprocessor::ReIDPreprocessor()
    : pool_manager_(nullptr)
    , vic_stream_(nullptr)
    , initialized_(false) {
}

ReIDPreprocessor::~ReIDPreprocessor() {
    release();
}

bool ReIDPreprocessor::initialize(dwContextHandle_t ctx, ReIDResourceManager* pool_manager) {
    if (initialized_) {
        return true;
    }

    if (!pool_manager) {
        std::cerr << "ReIDPreprocessor: Invalid pool manager" << std::endl;
        return false;
    }

    pool_manager_ = pool_manager;

    ReIdConditionerConfig config;
    config.engineHasPreprocess = false;

    if (!conditioner_.init(ctx, nullptr, &config)) {
        std::cerr << "ReIDPreprocessor: Failed to initialize conditioner" << std::endl;
        return false;
    }

    initialized_ = true;
    return true;
}

void ReIDPreprocessor::release() {
    if (!initialized_) {
        return;
    }

    conditioner_.release();
    initialized_ = false;
}

bool ReIDPreprocessor::prepareAsync(const ReIDRequest& request, uint32_t slot_id) {
    if (!initialized_) {
        return false;
    }

    TensorSlot* slot = pool_manager_->getTensorSlot(slot_id);
    if (!slot) {
        return false;
    }

    ReIDPreparedInput* prepared = &slot->prepared_input;
    prepared->slot_id = slot_id;
    prepared->camera_id = request.camera_id;
    prepared->local_track_id = request.local_track_id;
    prepared->timestamp_us = request.timestamp_us;
    prepared->source_bbox = request.bbox;
    prepared->release_callback = request.release_callback;
    prepared->user_data = request.user_data;

    dwDNNTensorHandle_t output_tensor = slot->input_tensor;
    if (!conditioner_.prepare(output_tensor, &request.image, 1, &request.bbox)) {
        std::cerr << "ReIDPreprocessor: VIC prepare failed for slot " << slot_id << std::endl;
        return false;
    }

    if (request.release_callback) {
        request.release_callback(request.image, request.user_data);
    }

    pool_manager_->pushPendingTask(slot_id);

    return true;
}

}
