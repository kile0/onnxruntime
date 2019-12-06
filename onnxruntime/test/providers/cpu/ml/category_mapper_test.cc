// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

template <typename TInput, typename TOutput>
static void RunTest(const Vector<int64_t>& dims, const Vector<TInput>& input, const Vector<TOutput>& output) {
  OpTester test("CategoryMapper", 1, onnxruntime::kMLDomain);

  static const AttributeVector<std::string> categories = {"Three", "Two", "One"};
  static const AttributeVector<int64_t> indexes = {3, 2, 1};

  test.AddAttribute("cats_strings", categories);
  test.AddAttribute("cats_int64s", indexes);

  test.AddAttribute("default_string", "default");
  test.AddAttribute<int64_t>("default_int64", 99);

  test.AddInput<TInput>("X", dims, input);
  test.AddOutput<TOutput>("Y", dims, output);

  test.Run();
}

TEST(CategoryMapper, StringToInt) {
  Vector<int64_t> dims{2, 2, 2};

  Vector<std::string> input{"Unknown", "Two", "Three", "B", "A", "One", "one", "two"};
  Vector<int64_t> output{99, 2, 3, 99, 99, 1, 99, 99};

  RunTest(dims, input, output);
}

TEST(CategoryMapper, IntToString) {
  Vector<int64_t> dims{2, 3};

  Vector<int64_t> input{1, 2, 3, 4, 5, 6};
  Vector<std::string> output{"One", "Two", "Three", "default", "default", "default"};

  RunTest(dims, input, output);
}
}  // namespace test
}  // namespace onnxruntime
