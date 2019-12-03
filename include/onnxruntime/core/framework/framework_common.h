// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include "run_options.h"

namespace onnxruntime {  // forward declarations
class Model;
class GraphTransformer;
class NodeArg;
}  // namespace onnxruntime

namespace onnxruntime {
using InputDefList = Vector<const onnxruntime::NodeArg*>;
using OutputDefList = Vector<const onnxruntime::NodeArg*>;

using NameMLValMap = std::unordered_map<std::string, OrtValue>;
}  // namespace onnxruntime
