// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <vector>
#include <atomic>
#include "core/session/inference_session.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/providers/providers.h"

struct OrtSessionOptions {
  onnxruntime::SessionOptions value;
  Vector<OrtCustomOpDomain*> custom_op_domains_;
  Vector<std::shared_ptr<onnxruntime::IExecutionProviderFactory>> provider_factories;
  OrtSessionOptions() = default;
  ~OrtSessionOptions();
  OrtSessionOptions(const OrtSessionOptions& other);
  OrtSessionOptions& operator=(const OrtSessionOptions& other);
};
