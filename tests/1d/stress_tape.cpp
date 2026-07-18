//
// Created by Pranav C on 18/07/2026.
//

//
// Stress test: a long chain of ops forces one heap allocation per node (per
// the vector<vector<double>> storage in values_one/grad_one), stressing the
// allocator and confirming backward_one stays correct.
#include <gtest/gtest.h>

#include "../../include/operations.h"
#include "../../include/tape.h"
#include "../../include/tensor_one.h"

using namespace autograd;

namespace autograd_tests {

TEST(Stress1DBackwardTest, LongChainManyAllocations) {
  constexpr size_t kTensorWidth = 256;
  constexpr size_t kChainLength = 50000;

  Tape tape(TensorClass::TENSOR_ONE, 1.0);

  TensorOne a(std::vector<double>(kTensorWidth, 1.0), true, &tape);
  TensorOne one(std::vector<double>(kTensorWidth, 1.0), false, &tape);

  TensorOne current = a;
  for (size_t i = 0; i < kChainLength; ++i) {
    current = current + one;
  }

  ASSERT_EQ(tape.nodes.size(), 2 + kChainLength);

  for (double v : current.data) {
    EXPECT_DOUBLE_EQ(v, 1.0 + static_cast<double>(kChainLength));
  }

  tape.backward_one(static_cast<int>(current.node_id));

  // Pure chain of adds: d(final)/da = 1 regardless of chain length.
  for (double g : tape.grad_one[a.node_id]) {
    EXPECT_DOUBLE_EQ(g, 1.0);
  }
}

}  // namespace autograd_tests