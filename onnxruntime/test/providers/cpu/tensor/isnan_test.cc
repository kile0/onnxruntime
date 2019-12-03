// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include <cmath>  // NAN
#include "core/util/math.h"

namespace onnxruntime {
namespace test {

TEST(IsNaNOpTest, IsNaNFloat) {
  OpTester test("IsNaN", 9, kOnnxDomain);
  Vector<int64_t> dims{2, 2};
  test.AddInput<float>("X", dims, {1.0f, NAN, 2.0f, NAN});
  test.AddOutput<bool>("Y", dims, {false, true, false, true});
  test.Run();
}

TEST(IsNaNOpTest, IsNaNFloat16) {
  OpTester test("IsNaN", 9, kOnnxDomain);
  Vector<int64_t> dims{2, 2};
  test.AddInput<MLFloat16>("X", dims, std::initializer_list<MLFloat16>({MLFloat16(math::floatToHalf(1.0f)), MLFloat16(math::floatToHalf(NAN)), MLFloat16(math::floatToHalf(2.0f)), MLFloat16(math::floatToHalf(NAN))}));
  test.AddOutput<bool>("Y", dims, {false, true, false, true});
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
