// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/graph/graph.h"

// forward declaration
struct OrtMemoryInfo;

namespace onnxruntime {
namespace nuphar {

bool NodeArgShapeUnknownOnAxis(const NodeArg* def, int64_t axis);

bool HasUnknownShapeOnAxis(const ConstPointerContainer<Vector<NodeArg*>>& defs, int64_t axis);

bool HasUnknownShapeOnAxes(const NodeArg* def, Vector<int64_t>& axes);

Status GetVectorInt64FromTensorProto(Vector<int64_t>& v,
                                     const ONNX_NAMESPACE::TensorProto& tp);

}  // namespace nuphar
}  // namespace onnxruntime
