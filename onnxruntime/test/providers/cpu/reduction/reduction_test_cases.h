// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {
namespace test {
struct ReductionAttribute {
  AttributeVector<int64_t> axes_;
  int64_t keep_dims_;
};

typedef std::tuple<ReductionAttribute, Vector<int64_t>, Vector<float>> OpAttributesResult;
typedef std::multimap<std::string, OpAttributesResult> OpAttributesResultMap;
struct ReductionTestCases {
  Vector<float> input_data;
  Vector<int64_t> input_dims;

  OpAttributesResultMap map_op_attribute_expected;
};

// python generated testcases
#include "reduction_test_cases.inl"

}  // namespace test
}  // namespace onnxruntime
