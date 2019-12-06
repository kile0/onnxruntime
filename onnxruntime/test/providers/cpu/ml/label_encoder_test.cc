// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

template <typename TInput, typename TOutput>
static void RunTest(const Vector<int64_t>& dims, const Vector<TInput>& input, const Vector<TOutput>& output) {
  OpTester test("LabelEncoder", 1, onnxruntime::kMLDomain);

  static const AttributeVector<std::string> labels = {"Beer", "Wine", "Tequila"};

  test.AddAttribute("classes_strings", labels);

  test.AddAttribute("default_string", "Water");
  test.AddAttribute<int64_t>("default_int64", 99);

  test.AddInput<TInput>("X", dims, input);
  test.AddOutput<TOutput>("Y", dims, output);

  test.Run();
}

TEST(LabelEncoder, StringToInt) {
  Vector<int64_t> dims{2, 2, 2};

  Vector<std::string> input{"Beer", "Burger", "Tequila", "Burrito", "Wine", "Cheese", "Tequila", "Floor"};
  Vector<int64_t> output{0, 99, 2, 99, 1, 99, 2, 99};

  RunTest(dims, input, output);
}

TEST(LabelEncoder, IntToString) {
  Vector<int64_t> dims{2, 3};

  Vector<int64_t> input{0, 10, 2, 3, 1, -1};
  Vector<std::string> output{"Beer", "Water", "Tequila", "Water", "Wine", "Water"};

  RunTest(dims, input, output);
}

TEST(LabelEncoder, StringToIntOpset2) {
  Vector<std::int64_t> dims{1, 5};

  Vector<std::string> input{"AA", "BB", "CC", "DD", "AA"};
  Vector<std::int64_t> output{9, 1, 5566, 4, 9};

  OpTester test("LabelEncoder", 2, onnxruntime::kMLDomain);

  const AttributeVector<std::string> keys{"AA", "BB", "DD"};
  const AttributeVector<std::int64_t> values{9, 1, 4};

  test.AddAttribute("keys_strings", keys);
  test.AddAttribute("values_int64s", values);
  test.AddAttribute("default_int64", (std::int64_t)5566);

  test.AddInput<std::string>("X", dims, input);
  test.AddOutput<std::int64_t>("Y", dims, output);

  test.Run();
}

TEST(LabelEncoder, IntToStringOpset2) {
  Vector<std::int64_t> dims{1, 5};

  Vector<std::int64_t> input{9, 1, 5566, 4, 9};
  Vector<std::string> output{"AA", "BB", "CC", "DD", "AA"};

  OpTester test("LabelEncoder", 2, onnxruntime::kMLDomain);

  const AttributeVector<std::int64_t> keys{9, 1, 4};
  const AttributeVector<std::string> values{"AA", "BB", "DD"};

  test.AddAttribute("keys_int64s", keys);
  test.AddAttribute("values_strings", values);
  test.AddAttribute<std::string>("default_string", "CC");

  test.AddInput<std::int64_t>("X", dims, input);
  test.AddOutput<std::string>("Y", dims, output);

  test.Run();
}

TEST(LabelEncoder, FloatToStringOpset2) {
  Vector<std::int64_t> dims{5, 1};

  Vector<float> input{9.4f, 1.7f, 3.6f, 1.2f, 2.8f};
  Vector<std::string> output{"AA", "BB", "DD", "CC", "CC"};

  OpTester test("LabelEncoder", 2, onnxruntime::kMLDomain);

  const AttributeVector<float> keys{9.4f, 1.7f, 3.6f};
  const AttributeVector<std::string> values{"AA", "BB", "DD"};

  test.AddAttribute("keys_floats", keys);
  test.AddAttribute("values_strings", values);
  test.AddAttribute<std::string>("default_string", "CC");

  test.AddInput<float>("X", dims, input);
  test.AddOutput<std::string>("Y", dims, output);

  test.Run();
}

TEST(LabelEncoder, StringToFloatOpset2) {
  Vector<std::int64_t> dims{5, 1};

  Vector<std::string> input{"AA", "BB", "DD", "CC", "CC"};
  Vector<float> output{9.4f, 1.7f, 3.6f, 55.66f, 55.66f};

  OpTester test("LabelEncoder", 2, onnxruntime::kMLDomain);

  const AttributeVector<std::string> keys{"AA", "BB", "DD"};
  const AttributeVector<float> values{9.4f, 1.7f, 3.6f};

  test.AddAttribute("keys_strings", keys);
  test.AddAttribute("values_floats", values);
  test.AddAttribute("default_float", 55.66f);

  test.AddInput<std::string>("X", dims, input);
  test.AddOutput<float>("Y", dims, output);

  test.Run();
}

TEST(LabelEncoder, FloatToInt64Opset2) {
  Vector<std::int64_t> dims{5};

  Vector<float> input{9.4f, 1.7f, 3.6f, 55.66f, 55.66f};
  Vector<std::int64_t> output{1, 9, 3, -8, -8};

  OpTester test("LabelEncoder", 2, onnxruntime::kMLDomain);

  const AttributeVector<float> keys{9.4f, 1.7f, 3.6f};
  const AttributeVector<std::int64_t> values{1, 9, 3};

  test.AddAttribute("keys_floats", keys);
  test.AddAttribute("values_int64s", values);
  test.AddAttribute("default_int64", (std::int64_t)-8);

  test.AddInput<float>("X", dims, input);
  test.AddOutput<std::int64_t>("Y", dims, output);

  test.Run();
}

TEST(LabelEncoder, Int64ToFloatOpset2) {
  Vector<std::int64_t> dims{5};

  Vector<std::int64_t> input{3, 1, 9, -8, -8};
  Vector<float> output{3.6f, 9.4f, 1.7f, 55.66f, 55.66f};

  OpTester test("LabelEncoder", 2, onnxruntime::kMLDomain);

  const AttributeVector<std::int64_t> keys{1, 9, 3};
  const AttributeVector<float> values{9.4f, 1.7f, 3.6f};

  test.AddAttribute("keys_int64s", keys);
  test.AddAttribute("values_floats", values);
  test.AddAttribute("default_float", 55.66f);

  test.AddInput<std::int64_t>("X", dims, input);
  test.AddOutput<float>("Y", dims, output);

  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
