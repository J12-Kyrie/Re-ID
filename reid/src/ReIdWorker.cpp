#include "ReIdWorker.h"
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <cstring>
#include <algorithm>

namespace reid {

bool ReIdWorker::init(dwContextHandle_t ctx, const ReIdWorkerConfig* config) {
    if (!ctx || !config || config->enginePath.empty()) return false;
    if (isInitialized()) return true;

    m_ctx = ctx;

    cudaError_t err = cudaStreamCreate(&m_stream);
    if (err != cudaSuccess) return false;

    if (!m_conditioner.init(ctx, m_stream, &config->conditionerConfig)) {
        cudaStreamDestroy(m_stream);
        m_stream = nullptr;
        return false;
    }

    if (!m_engine.init(config->enginePath.c_str(), config->dlaCore)) {
        m_conditioner.release();
        cudaStreamDestroy(m_stream);
        m_stream = nullptr;
        return false;
    }

    dwDNNTensorProperties inputProps;
    makeReIdInputProps(&inputProps);

    dwStatus status = dwDNNTensor_createWithExtMem(
        &m_inputTensor, &inputProps,
        static_cast<uint8_t*>(m_engine.dInput()), 0);

    if (status != DW_SUCCESS || m_inputTensor == DW_NULL_HANDLE) {
        m_engine.release();
        m_conditioner.release();
        cudaStreamDestroy(m_stream);
        m_stream = nullptr;
        return false;
    }

    return true;
}

void ReIdWorker::release() {
    if (m_inputTensor != DW_NULL_HANDLE) {
        dwDNNTensor_destroy(m_inputTensor);
        m_inputTensor = DW_NULL_HANDLE;
    }
    m_engine.release();
    m_conditioner.release();
    if (m_stream) {
        cudaStreamDestroy(m_stream);
        m_stream = nullptr;
    }
    m_ctx = DW_NULL_HANDLE;
}

bool ReIdWorker::process(dwImageHandle_t image, const dwRect& roi,
                         std::vector<float>* embedding) {
    if (!isInitialized() || !image || !embedding) return false;

    dwStatus status = m_conditioner.prepare(m_inputTensor, &image, 1, &roi);
    if (status != DW_SUCCESS) return false;

    if (!m_engine.infer(m_stream)) return false;

    cudaError_t err = cudaStreamSynchronize(m_stream);
    if (err != cudaSuccess) return false;

    embedding->resize(REID_OUTPUT_DIM);
    alignas(16) __half hFp16[REID_OUTPUT_DIM];

    if (!m_engine.copyOutputToHost(hFp16, sizeof(hFp16))) return false;

    for (uint32_t i = 0; i < REID_OUTPUT_DIM; ++i) {
        (*embedding)[i] = __half2float(hFp16[i]);
    }
    return true;
}

}  // namespace reid
