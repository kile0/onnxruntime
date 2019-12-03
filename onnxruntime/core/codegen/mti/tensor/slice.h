// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <string>
#include <tvm/tvm.h>

namespace onnxruntime {
namespace tvm_codegen {

tvm::Tensor Slice(const tvm::Tensor& X,
                  const Vector<int64_t>& starts,
                  const Vector<int64_t>& ends,
                  const Vector<int64_t>& axes,
                  const Vector<int64_t>& steps,
                  const std::string& name = "slice");

}  // namespace tvm_codegen
}  // namespace onnxruntime
