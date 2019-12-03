// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
namespace onnxruntime {
namespace ml {

class ZipMapOp final : public OpKernel {
 public:
  explicit ZipMapOp(const OpKernelInfo& info);
  common::Status Compute(OpKernelContext* context) const override;

 private:
  bool using_strings_;
  Vector<int64_t> classlabels_int64s_;
  Vector<std::string> classlabels_strings_;
};

}  // namespace ml
}  // namespace onnxruntime
