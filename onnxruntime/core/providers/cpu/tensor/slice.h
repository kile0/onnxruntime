// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"

namespace onnxruntime {

class SliceBase {
 protected:
  SliceBase(const OpKernelInfo& info, bool dynamic = false) {
    if (!dynamic) {
      auto has_starts = info.GetAttrs("starts", attr_starts_).IsOK();
      auto has_ends = info.GetAttrs("ends", attr_ends_).IsOK();
      auto has_axes = info.GetAttrs("axes", attr_axes_).IsOK();
      ORT_ENFORCE(has_starts && has_ends && attr_starts_.size() == attr_ends_.size(),
                  "Missing or invalid starts and ends attribute");
      ORT_ENFORCE(!has_axes || attr_axes_.size() == attr_starts_.size(),
                  "Invalid axes attribute, axes attribute (if present) should have the same size as starts/ends attributes");
    }
  }

  // compute output_dims without steps (Slice V1-9 & DynamicSlice)
  Status PrepareForCompute(const Vector<int64_t>& raw_starts,
                           const Vector<int64_t>& raw_ends,
                           const Vector<int64_t>& raw_axes,
                           const Vector<int64_t>& input_dimensions,
                           Vector<int64_t>& starts,
                           Vector<int64_t>& steps,
                           Vector<int64_t>& output_dims,
                           Vector<int64_t>*& flattened_output_dims) const;

  // compute output_dims with steps (Slice V10)
  Status PrepareForCompute(const Vector<int64_t>& raw_starts,
                           const Vector<int64_t>& raw_ends,
                           const Vector<int64_t>& raw_axes,
                           const Vector<int64_t>& raw_steps,
                           const Vector<int64_t>& input_dimensions,
                           Vector<int64_t>& starts,
                           Vector<int64_t>& steps,
                           Vector<int64_t>& output_dims,
                           Vector<int64_t>*& flattened_output_dims) const;

  // Slice V10 & DynamicSlice
  void FillVectorsFromInput(const OpKernelContext* context,
                            Vector<int64_t>& input_starts,
                            Vector<int64_t>& input_ends,
                            Vector<int64_t>& input_axes,
                            Vector<int64_t>& input_steps) const;

  Vector<int64_t> attr_starts_, attr_ends_, attr_axes_;
};

template <typename T, bool dynamic>
struct Slice final : public OpKernel, public SliceBase {
  Slice(const OpKernelInfo& info) : OpKernel(info), SliceBase(info, dynamic) {}
  Status Compute(OpKernelContext* context) const override;
};  // namespace onnxruntime

}  // namespace onnxruntime
