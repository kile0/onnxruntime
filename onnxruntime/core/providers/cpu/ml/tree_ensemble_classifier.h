// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "ml_common.h"

namespace onnxruntime {
namespace ml {
template <typename T>
class TreeEnsembleClassifier final : public OpKernel {
 public:
  explicit TreeEnsembleClassifier(const OpKernelInfo& info);
  common::Status Compute(OpKernelContext* context) const override;

 private:
  void Initialize();
  common::Status ProcessTreeNode(std::map<int64_t, float>& classes,
                                 int64_t treeindex,
                                 const T* x_data,
                                 int64_t feature_base) const;

  Vector<int64_t> nodes_treeids_;
  Vector<int64_t> nodes_nodeids_;
  Vector<int64_t> nodes_featureids_;
  Vector<float> nodes_values_;
  Vector<float> nodes_hitrates_;
  Vector<std::string> nodes_modes_names_;
  Vector<NODE_MODE> nodes_modes_;
  Vector<int64_t> nodes_truenodeids_;
  Vector<int64_t> nodes_falsenodeids_;
  Vector<int64_t> missing_tracks_true_;  // no bool type

  Vector<int64_t> class_nodeids_;
  Vector<int64_t> class_treeids_;
  Vector<int64_t> class_ids_;
  Vector<float> class_weights_;
  int64_t class_count_;
  std::set<int64_t> weights_classes_;

  Vector<float> base_values_;
  Vector<std::string> classlabels_strings_;
  Vector<int64_t> classlabels_int64s_;
  bool using_strings_;

  Vector<std::tuple<int64_t, int64_t, int64_t, float>> leafnodedata_;
  std::unordered_map<int64_t, int64_t> leafdata_map_;
  Vector<int64_t> roots_;
  const int64_t kOffset_ = 4000000000L;
  const int64_t kMaxTreeDepth_ = 1000;
  POST_EVAL_TRANSFORM post_transform_;
  bool weights_are_all_positive_;
};
}  // namespace ml
}  // namespace onnxruntime
