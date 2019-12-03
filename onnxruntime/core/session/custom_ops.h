// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

namespace onnxruntime {

common::Status CreateCustomRegistry(const Vector<OrtCustomOpDomain*>& op_domains, std::shared_ptr<CustomRegistry>& output);

}
