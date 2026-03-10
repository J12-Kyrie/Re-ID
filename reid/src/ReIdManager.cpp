#include "ReIdManager.h"
#include <algorithm>
#include <thread>

namespace reid {

bool ReIdManager::init(dwContextHandle_t ctx, const ReIdManagerConfig* config) {
    if (!ctx || !config) return false;
    if (isInitialized()) return true;

    ReIdWorkerConfig cfg0;
    cfg0.enginePath = config->enginePathDla0;
    cfg0.dlaCore   = 0;
    cfg0.conditionerConfig = config->conditionerConfig;

    ReIdWorkerConfig cfg1;
    cfg1.enginePath = config->enginePathDla1;
    cfg1.dlaCore   = 1;
    cfg1.conditionerConfig = config->conditionerConfig;

    if (!m_worker0.init(ctx, &cfg0)) return false;
    if (!m_worker1.init(ctx, &cfg1)) {
        m_worker0.release();
        return false;
    }
    return true;
}

void ReIdManager::release() {
    m_worker0.release();
    m_worker1.release();
}

bool ReIdManager::process(dwImageHandle_t image, const dwRect* rois,
                          uint32_t numRois,
                          std::vector<std::vector<float>>* embeddings) {
    if (!isInitialized() || !image || !rois || !embeddings) return false;

    embeddings->clear();
    embeddings->resize(numRois);

    for (uint32_t i = 0; i < numRois; i += 2) {
        bool ok0 = false, ok1 = false;
        std::thread t0([&]() { ok0 = m_worker0.process(image, rois[i], &(*embeddings)[i]); });
        if (i + 1 < numRois) {
            std::thread t1([&]() { ok1 = m_worker1.process(image, rois[i + 1], &(*embeddings)[i + 1]); });
            t0.join();
            t1.join();
            if (!ok0 || !ok1) return false;
        } else {
            t0.join();
            if (!ok0) return false;
        }
    }
    return true;
}

}  // namespace reid
