#ifndef REID_CONDITIONER_H
#define REID_CONDITIONER_H

#include <dw/core/context/Context.h>
#include <dw/dnn/dataconditioner/DataConditioner.h>
#include <dw/dnn/tensor/Tensor.h>
#include <dw/image/Image.h>

#include "ReIdTensorProps.h"

namespace reid {

struct ReIdConditionerConfig {
    bool engineHasPreprocess = false;  // if true, skip mean/std in conditioner
};

class ReIdConditioner {
public:
    ReIdConditioner() = default;
    ~ReIdConditioner() { release(); }

    ReIdConditioner(const ReIdConditioner&) = delete;
    ReIdConditioner& operator=(const ReIdConditioner&) = delete;

    dwStatus init(dwContextHandle_t ctx, cudaStream_t stream,
                  const ReIdConditionerConfig* config = nullptr);
    void release();

    dwStatus prepare(dwDNNTensorHandle_t tensorOutput,
                     dwImageHandle_t const* inputImages, uint32_t numImages,
                     dwRect const* rois);

    bool isInitialized() const { return m_handle != DW_NULL_HANDLE; }

private:
    dwDataConditionerHandle_t m_handle = DW_NULL_HANDLE;
    dwContextHandle_t m_ctx = DW_NULL_HANDLE;
};

}  // namespace reid

#endif  // REID_CONDITIONER_H
