//
// Created by Pranav C on 15/01/2026.
//

#include <gtest/gtest.h>

#include "../../include/operations.h"
#include "../../include/tape.h"
#include "../../include/tensor_one.h"

using namespace autograd;

namespace autograd_tests {

TEST(Basic1DTest, SimpleInstanceCheck) {
  Tape tape(TensorClass::TENSOR_ONE);
  TensorOne a(std::vector<double>{1.0, 2.0, 3.0}, true, &tape);
  TensorOne b(std::vector<double>{4.0, 5.0, 6.0}, true, &tape);

  EXPECT_EQ(a.size(), 3);
  EXPECT_EQ(b.size(), 3);
  EXPECT_EQ(tape.nodes.size(), 2);
}

TEST(Basic1DTest, BasicAdditionWithoutTape) {
  const TensorOne a(std::vector<double>{1.0, 2.0, 3.0});
  const TensorOne b(std::vector<double>{4.0, 5.0, 6.0});

  const TensorOne c = a + b;
  EXPECT_EQ(c.data[0], 5.0);
  EXPECT_EQ(c.data[1], 7.0);
  EXPECT_EQ(c.data[2], 9.0);
  EXPECT_EQ(c.tape, nullptr);
}

TEST(Basic1DTest, BasicSubtractionWithoutTape) {
  const TensorOne a(std::vector<double>{1.0, 2.0, 3.0});
  const TensorOne b(std::vector<double>{4.0, 5.0, 6.0});

  const TensorOne c = a - b;
  EXPECT_EQ(c.data[0], -3.0);
  EXPECT_EQ(c.data[1], -3.0);
  EXPECT_EQ(c.data[2], -3.0);
  EXPECT_EQ(c.tape, nullptr);
}

TEST(Basic1DTest, BasicMultiplicationWithoutTape) {
  const TensorOne a(std::vector<double>{1.0, 2.0, 3.0});
  const TensorOne b(std::vector<double>{4.0, 5.0, 6.0});

  const TensorOne c = a * b;
  EXPECT_EQ(c.data[0], 4.0);
  EXPECT_EQ(c.data[1], 10.0);
  EXPECT_EQ(c.data[2], 18.0);
  EXPECT_EQ(c.tape, nullptr);
}

}  // namespace autograd_tests
