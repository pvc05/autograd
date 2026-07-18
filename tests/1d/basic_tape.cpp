//
// Created by Pranav C on 07/06/2026.
//
#include <gtest/gtest.h>

#include "../../include/operations.h"
#include "../../include/tape.h"
#include "../../include/tensor_one.h"

using namespace autograd;

namespace autograd_tests {

TEST(Basic1DTest, OperationWithTape) {
  Tape tape(TensorClass::TENSOR_ONE);
  TensorOne a(std::vector<double>{1.0, 2.0, 3.0}, true, &tape);
  TensorOne b(std::vector<double>{4.0, 5.0, 6.0}, true, &tape);

  TensorOne c = a + b;  // new node
  TensorOne d = c * a;  // new node
  TensorOne e = d - b;  // new node

  EXPECT_EQ(e.data[0], (1.0 + 4.0) * 1.0 - 4.0);
  EXPECT_EQ(e.data[1], (2.0 + 5.0) * 2.0 - 5.0);
  EXPECT_EQ(e.data[2], (3.0 + 6.0) * 3.0 - 6.0);

  // Constants (2) + add (1) + mul (1) + sub (1) = 5 nodes
  EXPECT_EQ(tape.nodes.size(), 5);
}

TEST(Basic1DTest, OperationWithTapeSelf) {
  Tape tape(TensorClass::TENSOR_ONE);
  TensorOne a(std::vector<double>{1.0, 2.0, 3.0}, true, &tape);

  TensorOne b = a + a;
  TensorOne c = b * a;

  EXPECT_EQ(c.data[0], 2.0);
  EXPECT_EQ(c.data[1], 8.0);
  EXPECT_EQ(c.data[2], 18.0);

  EXPECT_EQ(tape.nodes.size(), 3);
}
}  // namespace autograd_tests
