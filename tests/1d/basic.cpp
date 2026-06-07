//
// Created by Pranav C on 15/01/2026.
//

#include <gtest/gtest.h>

#include "tape.h"
#include "tensor_one.h"

using namespace autograd;

namespace autograd_tests {

TEST(Basic1DTest, SimpleInstanceCheck) {
  // Create a tape for 1D tensors
  Tape tape(TENSOR_ONE);

  // Create two 1D tensors
  TensorOne tensor_a({1.0, 2.0, 3.0}, true, &tape);
  TensorOne tensor_b({4.0, 5.0, 6.0}, true, &tape);
}

}  // namespace autograd_tests
