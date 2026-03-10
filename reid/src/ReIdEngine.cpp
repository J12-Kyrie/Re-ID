#include "ReIdEngine.h"
#include <cuda_runtime_api.h>
#include <fstream>
#include <algorithm>
#include <cstring>

namespace reid {

namespace {

class TrtLogger : public nvinfer1::ILogger {
public:
    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override {
        if (severity <= nvinfer1::ILogger::Severity::kWARNING) {
            // Suppress INFO/VERBOSE
        }
    }
};

static TrtLogger g_logger;

}  // namespace

bool ReIdEngine::init(const char* enginePath, uint32_t dlaCore) {
    if (m_context) return true;

    std::ifstream f(enginePath, std::ios::binary);
    if (!f) return false;

    std::vector<char> data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    f.close();
    if (data.empty()) return false;

    m_runtime.reset(nvinfer1::createInferRuntime(g_logger));
    if (!m_runtime) return false;

    m_runtime->setDLACore(static_cast<int>(dlaCore));

    m_engine.reset(m_runtime->deserializeCudaEngine(data.data(), data.size()));
    if (!m_engine) return false;

    m_context.reset(m_engine->createExecutionContext());
    if (!m_context) return false;

    const int nb = m_engine->getNbIOTensors();
    for (int i = 0; i < nb; ++i) {
        nvinfer1::Dims dims = m_engine->getTensorShape(m_engine->getIOTensorName(i));
        if (dims.nbDims == 4 && dims.d[2] == 3) {
            m_inputBindingIdx = i;
        } else {
            m_outputBindingIdx = i;
        }
    }
    if (m_inputBindingIdx < 0 || m_outputBindingIdx < 0) return false;

    cudaError_t err;
    err = cudaMalloc(&m_dInput, REID_INPUT_BYTES);
    if (err != cudaSuccess) return false;

    err = cudaMalloc(&m_dOutput, REID_OUTPUT_BYTES);
    if (err != cudaSuccess) {
        cudaFree(m_dInput);
        m_dInput = nullptr;
        return false;
    }

    m_context->setTensorAddress(m_engine->getIOTensorName(m_inputBindingIdx), m_dInput);
    m_context->setTensorAddress(m_engine->getIOTensorName(m_outputBindingIdx), m_dOutput);

    return true;
}

void ReIdEngine::release() {
    if (m_dInput) {
        cudaFree(m_dInput);
        m_dInput = nullptr;
    }
    if (m_dOutput) {
        cudaFree(m_dOutput);
        m_dOutput = nullptr;
    }
    m_context.reset();
    m_engine.reset();
    m_runtime.reset();
    m_inputBindingIdx = m_outputBindingIdx = -1;
}

bool ReIdEngine::infer(cudaStream_t stream) {
    if (!m_context) return false;
    return m_context->enqueueV3(stream);
}

bool ReIdEngine::copyOutputToHost(void* hOutput, size_t outputBytes) {
    if (!m_dOutput || !hOutput || outputBytes < REID_OUTPUT_BYTES) return false;
    return cudaMemcpy(hOutput, m_dOutput, REID_OUTPUT_BYTES, cudaMemcpyDeviceToHost) == cudaSuccess;
}

}  // namespace reid
