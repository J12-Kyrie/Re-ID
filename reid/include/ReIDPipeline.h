#ifndef REID_PIPELINE_H
#define REID_PIPELINE_H

#include "ReIDTypes.h"
#include "ReIDResourceManager.h"
#include "ReIDPreprocessor.h"
#include "ReIDWorkerPool.h"
#include "GlobalAssociation.h"

namespace reid {

class ReIDPipeline {
public:
    ReIDPipeline();
    ~ReIDPipeline();

    ReIDPipeline(const ReIDPipeline&) = delete;
    ReIDPipeline& operator=(const ReIDPipeline&) = delete;

    bool initialize(dwContextHandle_t ctx, const ReIDPipelineConfig& config);
    void shutdown();

    // Simple YOLO integration API
    bool submitROI(const ReIDRequest& request);

    size_t pollResults(ReIDResult* results, size_t maxResults);

    PipelineStats getStats() const;

    bool isInitialized() const { return initialized_; }

private:
    ReIDResourceManager resource_manager_;
    ReIDPreprocessor preprocessor_;
    ReIDWorkerPool worker_pool_;
    GlobalAssociation global_association_;

    dwContextHandle_t ctx_;
    bool initialized_;
};

}

#endif
