#ifndef REID_WORKER_POOL_H
#define REID_WORKER_POOL_H

#include "ReIDTypes.h"
#include "ReIDResourceManager.h"
#include "ReIdWorker.h"
#include <thread>
#include <atomic>
#include <vector>

namespace reid {

class ReIDWorkerPool {
public:
    ReIDWorkerPool();
    ~ReIDWorkerPool();

    ReIDWorkerPool(const ReIDWorkerPool&) = delete;
    ReIDWorkerPool& operator=(const ReIDWorkerPool&) = delete;

    bool initialize(dwContextHandle_t ctx,
                   const char* engine_path_dla0,
                   const char* engine_path_dla1,
                   ReIDResourceManager* pool_manager);
    void release();

    void start();
    void stop();

    bool isInitialized() const { return initialized_; }

private:
    void workerThreadLoop(int dla_core_id);

    ReIdWorker worker0_;
    ReIdWorker worker1_;

    ReIDResourceManager* pool_manager_;

    std::thread worker_thread0_;
    std::thread worker_thread1_;

    std::atomic<bool> running_;
    bool initialized_;
};

}

#endif
