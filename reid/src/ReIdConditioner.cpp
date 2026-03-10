#include "ReIdConditioner.h"
#include <cuda_runtime_api.h>
#include <algorithm>

namespace reid {

// ImageNet mean/std for resnet50_market1501 (in [0,255] for DataConditioner)
static const float MEAN_RGB[3] = {123.675f, 116.28f, 103.53f};   // 0.485,0.456,0.406 * 255
static const float STD_RGB[3]  = {58.395f, 57.12f, 57.375f};     // 0.229,0.224,0.225 * 255

dwStatus ReIdConditioner::init(dwContextHandle_t ctx, cudaStream_t stream,
                               const ReIdConditionerConfig* config) {
    if (m_handle != DW_NULL_HANDLE) return DW_SUCCESS;
    if (ctx == DW_NULL_HANDLE) return DW_INVALID_HANDLE;

    m_ctx = ctx;

    dwDNNTensorProperties inputProps;
    makeReIdInputProps(&inputProps);

    dwDataConditionerParams params;
    dwDataConditioner_initParams(&params);

    params.splitPlanes        = true;
    params.ignoreAspectRatio  = true;
    params.scaleCoefficient  = 1.0f;
    params.meanImage         = nullptr;
    params.doPerPlaneMeanNormalization = false;
    params.convertToGray     = false;

    if (config && config->engineHasPreprocess) {
        params.meanValue[0] = params.meanValue[1] = params.meanValue[2] = 0.0f;
        params.stdev[0] = params.stdev[1] = params.stdev[2] = 1.0f;
    } else {
        params.meanValue[0] = MEAN_RGB[0];
        params.meanValue[1] = MEAN_RGB[1];
        params.meanValue[2] = MEAN_RGB[2];
        params.stdev[0] = STD_RGB[0];
        params.stdev[1] = STD_RGB[1];
        params.stdev[2] = STD_RGB[2];
    }

    dwStatus status = dwDataConditioner_initializeFromTensorProperties(
        &m_handle, &inputProps, 1U, &params, stream, ctx);

    if (status == DW_SUCCESS) {
        dwDataConditioner_setCUDAStream(stream, m_handle);
    }
    return status;
}

void ReIdConditioner::release() {
    if (m_handle != DW_NULL_HANDLE) {
        dwDataConditioner_release(m_handle);
        m_handle = DW_NULL_HANDLE;
    }
    m_ctx = DW_NULL_HANDLE;
}

dwStatus ReIdConditioner::prepare(dwDNNTensorHandle_t tensorOutput,
                                  dwImageHandle_t const* inputImages,
                                  uint32_t numImages, dwRect const* rois) {
    if (m_handle == DW_NULL_HANDLE || !tensorOutput || !inputImages || !rois) {
        return DW_INVALID_ARGUMENT;
    }
    return dwDataConditioner_prepareData(tensorOutput, inputImages, numImages,
                                         rois, cudaAddressModeClamp, m_handle);
}

}  // namespace reid
