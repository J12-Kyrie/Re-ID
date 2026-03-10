#ifndef REID_PREPROCESSOR_H
#define REID_PREPROCESSOR_H

#include "ReIDTypes.h"
#include "ReIDResourceManager.h"
#include "ReIdConditioner.h"

namespace reid {

class ReIDPreprocessor {
public:
    ReIDPreprocessor();
    ~ReIDPreprocessor();

    ReIDPreprocessor(const ReIDPreprocessor&) = delete;
    ReIDPreprocessor& operator=(const ReIDPreprocessor&) = delete;

    bool initialize(dwContextHandle_t ctx, ReIDResourceManager* pool_manager);
    void release();

    bool prepareAsync(const ReIDRequest& request, uint32_t slot_id);

    bool isInitialized() const { return initialized_; }

private:
    ReIdConditioner conditioner_;
    ReIDResourceManager* pool_manager_;
    void* vic_stream_;
    bool initialized_;
};

}

#endif
