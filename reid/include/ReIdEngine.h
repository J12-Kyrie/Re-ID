#ifndef REID_ENGINE_H
#define REID_ENGINE_H

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "ReIdTensorProps.h"

namespace reid {

class ReIdEngine {
public:
    ReIdEngine() = default;
    ~ReIdEngine() { release(); }

    ReIdEngine(const ReIdEngine&) = delete;
    ReIdEngine& operator=(const ReIdEngine&) = delete;

    bool init(const char* enginePath, uint32_t dlaCore);
    void release();

    bool infer(cudaStream_t stream);
    bool copyOutputToHost(void* hOutput, size_t outputBytes);

    void* dInput() { return m_dInput; }
    void* dOutput() { return m_dOutput; }
    bool isInitialized() const { return m_context != nullptr; }

private:
    std::unique_ptr<nvinfer1::IRuntime> m_runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> m_engine;
    std::unique_ptr<nvinfer1::IExecutionContext> m_context;

    void* m_dInput  = nullptr;
    void* m_dOutput = nullptr;

    int m_inputBindingIdx  = -1;
    int m_outputBindingIdx = -1;
};

}  // namespace reid

#endif  // REID_ENGINE_H
