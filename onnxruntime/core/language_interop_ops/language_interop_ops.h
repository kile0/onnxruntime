// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <string>
#include <vector>
#include <memory>
#include "core/graph/graph.h"
#include "core/session/onnxruntime_cxx_api.h"

namespace onnxruntime {
void InterOpDomainDeleter(OrtCustomOpDomain*);
using InterOpLogFunc = std::function<void(const char*)>;
using InterOpDomains = Vector<std::unique_ptr<OrtCustomOpDomain,decltype(&InterOpDomainDeleter)>>;
void LoadInterOp(const std::basic_string<ORTCHAR_T>& model_uri, InterOpDomains& domains, const InterOpLogFunc& log_func);
void LoadInterOp(const ONNX_NAMESPACE::ModelProto& model_proto, InterOpDomains& domains, const InterOpLogFunc& log_func);
void LoadInterOp(const ONNX_NAMESPACE::GraphProto& graph_proto, InterOpDomains& domains, const InterOpLogFunc& log_func);
}
