#include "ReIDPipeline.h"
#include <iostream>

using namespace reid;

int main() {
    std::cout << "=== Re-ID Pipeline Usage Example ===" << std::endl;

    dwContextHandle_t ctx = nullptr;

    ReIDPipelineConfig config;
    config.engine_path_dla0 = "reid_resnet50_dla0.engine";
    config.engine_path_dla1 = "reid_resnet50_dla1.engine";
    config.pool_size = 128;
    config.enable_global_association = true;

    ReIDPipeline pipeline;
    if (!pipeline.initialize(ctx, config)) {
        std::cerr << "Failed to initialize pipeline" << std::endl;
        return -1;
    }

    std::cout << "\n=== Simulating YOLO Detection ===" << std::endl;

    for (int frame = 0; frame < 10; ++frame) {
        dwImageHandle_t image = nullptr;

        for (int person = 0; person < 3; ++person) {
            ReIDRequest req;
            req.image = image;
            req.bbox = {100 + person * 50, 100, 64, 128};
            req.local_track_id = frame * 10 + person;
            req.camera_id = 0;
            req.timestamp_us = frame * 33333;
            req.release_callback = nullptr;
            req.user_data = nullptr;

            if (pipeline.submitROI(req)) {
                std::cout << "Frame " << frame << ": Submitted track_id "
                         << req.local_track_id << std::endl;
            } else {
                std::cout << "Frame " << frame << ": Pool exhausted!" << std::endl;
            }
        }
    }

    std::cout << "\n=== Polling Results ===" << std::endl;

    ReIDResult results[16];
    size_t count = pipeline.pollResults(results, 16);

    std::cout << "Retrieved " << count << " results" << std::endl;
    for (size_t i = 0; i < count; ++i) {
        std::cout << "  Result " << i << ": track_id=" << results[i].local_track_id
                 << ", camera=" << (int)results[i].camera_id
                 << ", global_id=" << results[i].global_track_id << std::endl;
    }

    std::cout << "\n=== Pipeline Statistics ===" << std::endl;
    PipelineStats stats = pipeline.getStats();
    std::cout << "Free slots: " << stats.free_slots << std::endl;
    std::cout << "Pending tasks: " << stats.pending_tasks << std::endl;
    std::cout << "Completed tasks: " << stats.completed_tasks << std::endl;
    std::cout << "Total submitted: " << stats.total_submitted << std::endl;
    std::cout << "Total completed: " << stats.total_completed << std::endl;
    std::cout << "Total dropped: " << stats.total_dropped << std::endl;

    pipeline.shutdown();

    std::cout << "\n=== Example Complete ===" << std::endl;

    return 0;
}
