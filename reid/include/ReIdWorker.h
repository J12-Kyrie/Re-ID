#ifndef REID_WORKER_H
#define REID_WORKER_H

#include <dw/core/context/Context.h>
#include <dw/image/Image.h>

#include <string>
#include <vector>

#include "ReIdConditioner.h"
#include "ReIdEngine.h"
#include "ReIdTensorProps.h"

namespace reid {

struct ReIdWorkerConfig {
    std::string enginePath;
    uint32_t dlaCore = 0;
    ReIdConditionerConfig conditionerConfig;
};

class ReIdWorker {
public:
    ReIdWorker() = default;
    ~ReIdWorker() { release(); }

    ReIdWorker(const ReIdWorker&) = delete;
    ReIdWorker& operator=(const ReIdWorker&) = delete;

    bool init(dwContextHandle_t ctx, const ReIdWorkerConfig* config);
    void release();

    bool process(dwImageHandle_t image, const dwRect& roi, std::vector<float>* embedding);

    bool isInitialized() const {
        return m_conditioner.isInitialized() && m_engine.isInitialized();
    }

private:
    dwContextHandle_t m_ctx       = DW_NULL_HANDLE;
    cudaStream_t m_stream        = nullptr;
    ReIdConditioner m_conditioner;
    ReIdEngine m_engine;
    dwDNNTensorHandle_t m_inputTensor = DW_NULL_HANDLE;
};

}  // namespace reid

#endif  // REID_WORKER_H
