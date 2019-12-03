// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"
#include "ml_common.h"

namespace onnxruntime {
namespace ml {

template <typename T>
class LinearRegressor final : public OpKernel {
 public:
  LinearRegressor(const OpKernelInfo& info);
  Status Compute(OpKernelContext* context) const override;

 private:
  int64_t targets_;
  Vector<float> coefficients_;
  Vector<float> intercepts_;
  POST_EVAL_TRANSFORM post_transform_;
};

}  // namespace ml
}  // namespace onnxruntime
