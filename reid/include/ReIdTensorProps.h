#ifndef REID_TENSOR_PROPS_H
#define REID_TENSOR_PROPS_H

#include <dw/core/base/GeometricTypes.h>
#include <dw/dnn/tensor/Tensor.h>

namespace reid {

// Re-ID engine input: [1, 3, 256, 128] FP16 NCHW
// dimensionSize order for NCHW: [0]=W, [1]=H, [2]=C, [3]=N
constexpr uint32_t REID_INPUT_W = 128;
constexpr uint32_t REID_INPUT_H = 256;
constexpr uint32_t REID_INPUT_C = 3;
constexpr uint32_t REID_INPUT_N = 1;

// Re-ID engine output: [1, 256] FP16
constexpr uint32_t REID_OUTPUT_DIM = 256;
constexpr uint32_t REID_OUTPUT_N = 1;

// Byte sizes for cudaMalloc
constexpr size_t REID_INPUT_BYTES =
    REID_INPUT_N * REID_INPUT_C * REID_INPUT_H * REID_INPUT_W * 2;  // sizeof(__half)=2
constexpr size_t REID_OUTPUT_BYTES = REID_OUTPUT_N * REID_OUTPUT_DIM * 2;

inline void makeReIdInputProps(dwDNNTensorProperties* props) {
    if (!props) return;
    props->dataType       = DW_TYPE_FLOAT16;
    props->tensorType     = DW_DNN_TENSOR_TYPE_CUDA;
    props->tensorLayout   = DW_DNN_TENSOR_LAYOUT_NCHW;
    props->isGPUMapped   = false;
    props->numDimensions = 4U;
    props->dimensionSize[0] = REID_INPUT_W;
    props->dimensionSize[1] = REID_INPUT_H;
    props->dimensionSize[2] = REID_INPUT_C;
    props->dimensionSize[3] = REID_INPUT_N;
    props->colorSpace = DW_DNN_TENSOR_COLORSPACE_RGB;
}

inline void makeReIdOutputProps(dwDNNTensorProperties* props) {
    if (!props) return;
    props->dataType       = DW_TYPE_FLOAT16;
    props->tensorType     = DW_DNN_TENSOR_TYPE_CUDA;
    props->tensorLayout   = DW_DNN_TENSOR_LAYOUT_NCHW;
    props->isGPUMapped   = false;
    props->numDimensions = 2U;
    props->dimensionSize[0] = REID_OUTPUT_DIM;
    props->dimensionSize[1] = REID_OUTPUT_N;
    props->colorSpace = DW_DNN_TENSOR_COLORSPACE_UNKNOWN;
}

inline dwRect makeDwRect(int32_t x, int32_t y, int32_t width, int32_t height) {
    dwRect r{};
    r.x      = x;
    r.y      = y;
    r.width  = width > 0 ? width : 1;
    r.height = height > 0 ? height : 1;
    return r;
}

}  // namespace reid

#endif  // REID_TENSOR_PROPS_H
