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
class LinearClassifier final : public OpKernel {
 public:
  LinearClassifier(const OpKernelInfo& info);
  Status Compute(OpKernelContext* context) const override;

 private:
  int64_t multi_class_;
  int64_t class_count_;
  POST_EVAL_TRANSFORM post_transform_;
  bool using_strings_;
  Vector<float> coefficients_;
  Vector<float> intercepts_;
  Vector<std::string> classlabels_strings_;
  Vector<int64_t> classlabels_ints_;
};

}  // namespace ml
}  // namespace onnxruntime
