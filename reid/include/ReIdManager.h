#ifndef REID_MANAGER_H
#define REID_MANAGER_H

#include <dw/core/context/Context.h>
#include <dw/image/Image.h>

#include <string>
#include <vector>

#include "ReIdWorker.h"

namespace reid {

struct ReIdManagerConfig {
    std::string enginePathDla0;
    std::string enginePathDla1;
    ReIdConditionerConfig conditionerConfig;
};

class ReIdManager {
public:
    ReIdManager() = default;
    ~ReIdManager() { release(); }

    ReIdManager(const ReIdManager&) = delete;
    ReIdManager& operator=(const ReIdManager&) = delete;

    bool init(dwContextHandle_t ctx, const ReIdManagerConfig* config);
    void release();

    bool process(dwImageHandle_t image, const dwRect* rois, uint32_t numRois,
                 std::vector<std::vector<float>>* embeddings);

    bool isInitialized() const {
        return m_worker0.isInitialized() && m_worker1.isInitialized();
    }

private:
    ReIdWorker m_worker0;
    ReIdWorker m_worker1;
};

}  // namespace reid

#endif  // REID_MANAGER_H
