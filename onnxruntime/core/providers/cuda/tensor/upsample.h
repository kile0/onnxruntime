// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cpu/tensor/upsample.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
class Upsample : public UpsampleBase, public CudaKernel {
 public:
  Upsample(OpKernelInfo info) : UpsampleBase(info), CudaKernel(info) {
  }

  Status ComputeInternal(OpKernelContext* context) const override;
  Status BaseCompute(OpKernelContext* context, const Vector<float>& roi, const Vector<float>& scales,
                     const Vector<int64_t>& output_dims) const;
};

}  // namespace cuda
}  // namespace onnxruntime
