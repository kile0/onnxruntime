// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>

#include "core/util/math.h"
#include "core/mlas/inc/mlas.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

using namespace onnxruntime::test;

namespace onnxruntime {
namespace test {

TEST(MathOpTest, AffineDefaultAttributes) {
  OpTester test("Affine");
  Vector<int64_t> dims{2, 2};
  test.AddInput<float>("A", dims, {0.0f, 1.0f, 2.0f, 3.0f});
  test.AddOutput<float>("B", dims, {0.0f, 1.0f, 2.0f, 3.0f});
  test.Run();
}

TEST(MathOpTest, Affine) {
  OpTester test("Affine");
  Vector<int64_t> dims{2, 2};
  test.AddAttribute("alpha", 2.0f);
  test.AddAttribute("beta", 1.0f);
  test.AddInput<float>("A", dims, {0.0f, 1.0f, 2.0f, 3.0f});
  test.AddOutput<float>("B", dims, {1.0f, 3.0f, 5.0f, 7.0f});
  test.Run();
}

TEST(MathOpTest, Scale) {
  OpTester test("Scale");
  Vector<int64_t> dims{2, 2};
  test.AddAttribute("scale", 2.0f);
  test.AddInput<float>("A", dims, {0.0f, 1.0f, 2.0f, 3.0f});
  test.AddOutput<float>("B", dims, {0.0f, 2.0f, 4.0f, 6.0f});
  test.Run();
}

TEST(MathOpTest, Scale_Default) {
  OpTester test("Scale");
  Vector<int64_t> dims{2, 2};
  test.AddInput<float>("A", dims, {0.0f, 1.0f, 2.0f, 3.0f});
  test.AddOutput<float>("B", dims, {0.0f, 1.0f, 2.0f, 3.0f});
  test.Run();
}

Vector<float> Add_Simple(const Vector<float>& input_a_data, const Vector<float>& input_b_data) {
  EXPECT_TRUE(input_a_data.size() % input_b_data.size() == 0 || input_b_data.size() % input_a_data.size() == 0);
  const Vector<float>& input_large_size = input_a_data.size() >= input_b_data.size() ? input_a_data : input_b_data;
  const Vector<float>& input_small_size = input_a_data.size() < input_b_data.size() ? input_a_data : input_b_data;

  Vector<float> output(input_large_size.size());
  for (size_t iter = 0; iter < input_large_size.size() / input_small_size.size(); iter++) {
    std::transform(input_large_size.begin() + iter * input_small_size.size(),
                   input_large_size.begin() + (iter + 1) * input_small_size.size(),
                   input_small_size.begin(),
                   output.begin() + iter * input_small_size.size(),
                   [](float a, float b) {
                     return a + b;
                   });
  }
  return output;
}

const Vector<float> ComputeGeluWithErf(const Vector<float>& input_data) {
  Vector<float> output(input_data.size());

  std::transform(input_data.begin(),
                 input_data.end(),
                 output.begin(),
                 [](float x) {
                   float y = erf(x * static_cast<float>(M_SQRT1_2));
                   return x * 0.5f * (y + 1.0f);
                 });

  return output;
}

static void RunBiasGeluTest(
    const Vector<float>& input_a_data,
    const Vector<float>& input_b_data,
    const Vector<int64_t>& input_a_dims,
    const Vector<int64_t>& input_b_dims) {
  if (HasCudaEnvironment(0)) {
    Vector<float> output_data = ComputeGeluWithErf(Add_Simple(input_a_data, input_b_data));

    OpTester tester("BiasGelu", 1, onnxruntime::kMSDomain);

    const Vector<int64_t>& output_dims = input_a_dims.size() >= input_b_dims.size() ? input_a_dims : input_b_dims;
    tester.AddInput<float>("A", input_a_dims, input_a_data);
    tester.AddInput<float>("B", input_b_dims, input_b_data);
    tester.AddOutput<float>("C", output_dims, output_data);

    tester.Run();
  }
}

TEST(BiasGeluTest, Two_One_Dim) {
  Vector<float> input_a_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  Vector<float> input_b_data = {
      -0.5f, 0.6f, 1.2f, 2.1f};

  RunBiasGeluTest(input_a_data, input_b_data, {2, 4}, {4});
}

}  // namespace test
}  // namespace onnxruntime
