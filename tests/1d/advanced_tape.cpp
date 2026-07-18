//
// Created by Pranav C on 18/07/2026.
//
//
// Edge-case tests probing the limits of the 1D reverse-mode backward pass.
//
#include <gtest/gtest.h>

#include "../../include/operations.h"
#include "../../include/tape.h"
#include "../../include/tensor_one.h"

using namespace autograd;

namespace autograd_tests {

// c = a + a: both parent slots point at the same node, so the gradient must
// accumulate twice, not overwrite.
TEST(Advanced1DBackwardTest, SelfReferentialDoubleCount) {
  Tape tape(TensorClass::TENSOR_ONE, 1.0);
  const TensorOne a(std::vector<double>{3.0, 4.0, 5.0}, true, &tape);

  const TensorOne c = a + a;
  tape.backward_one(static_cast<int>(c.node_id));

  for (double g : tape.grad_one[a.node_id]) {
    EXPECT_DOUBLE_EQ(g, 2.0);
  }
}

// b = a+a, c = a*a, d = b+c: a feeds the output through two distinct
// branches that recombine. dd/da = 2 + 2a (sum-over-paths rule).
TEST(Advanced1DBackwardTest, DiamondMultiPathAccumulation) {
  Tape tape(TensorClass::TENSOR_ONE, 1.0);
  TensorOne a(std::vector<double>{1.0, 2.0, 3.0}, true, &tape);

  TensorOne b = a + a;
  TensorOne c = a * a;
  TensorOne d = b + c;

  tape.backward_one(static_cast<int>(d.node_id));

  const std::vector<double> grad_a = tape.grad_one[a.node_id];
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_DOUBLE_EQ(grad_a[i], 2.0 + 2.0 * a.data[i]);
  }
}

// z has requires_grad=false but shares an ADD with p, which does require
// grad. The current backward_one writes into BOTH parent slots unconditionally
// when processing an ADD node, required_grads only gates whether a node's
// own turn in the reverse loop runs, not whether an individual parent write
// happens. So z's grad slot ends up populated too, even though z was declared
// as not requiring grad. This documents that behavior rather than assuming
// PyTorch-style "no grad allocated" semantics.
TEST(Advanced1DBackwardTest, RequiresGradIsNotPerOperandGated) {
  Tape tape(TensorClass::TENSOR_ONE, 1.0);
  TensorOne z(std::vector<double>{10.0, 20.0, 30.0}, false, &tape);
  TensorOne p(std::vector<double>{1.0, 2.0, 3.0}, true, &tape);

  TensorOne w = z + p;
  tape.backward_one(static_cast<int>(w.node_id));

  for (double g : tape.grad_one[p.node_id]) {
    EXPECT_DOUBLE_EQ(g, 1.0);
  }
  // Documenting current behavior: z's slot is NOT left at 0.
  for (double g : tape.grad_one[z.node_id]) {
    EXPECT_DOUBLE_EQ(g, 1.0);
  }
}

// backward_one's zero-clearing pass zeroes grad_one[0..node_id] on every
// call. If you reuse one tape for two independent computations and call
// backward_one on the second (later) output, that zero pass sweeps over the
// FIRST computation's node range too and wipes its already-read-out
// gradients, despite logical unrelation. This is a real
// gotcha for shared-tape reuse, not just a hypothetical.
TEST(Advanced1DBackwardTest, SecondBackwardCallClobbersEarlierResults) {
  Tape tape(TensorClass::TENSOR_ONE, 1.0);

  TensorOne p1(std::vector<double>{2.0, 3.0}, true, &tape);
  TensorOne p2(std::vector<double>{4.0, 5.0}, true, &tape);
  TensorOne out1 = p1 * p2;

  tape.backward_one(static_cast<int>(out1.node_id));
  EXPECT_DOUBLE_EQ(tape.grad_one[p1.node_id][0], 4.0);  // = p2
  EXPECT_DOUBLE_EQ(tape.grad_one[p2.node_id][0], 2.0);  // = p1

  TensorOne q1(std::vector<double>{10.0, 10.0}, true, &tape);
  TensorOne q2(std::vector<double>{1.0, 1.0}, true, &tape);
  TensorOne out2 = q1 + q2;

  tape.backward_one(static_cast<int>(out2.node_id));
  EXPECT_DOUBLE_EQ(tape.grad_one[q1.node_id][0], 1.0);
  EXPECT_DOUBLE_EQ(tape.grad_one[q2.node_id][0], 1.0);

  // p1's earlier gradient is gone, clobbered by the second call's zero pass.
  EXPECT_DOUBLE_EQ(tape.grad_one[p1.node_id][0], 0.0);
}

// A 5-op chain mixing add/mul/sub, verified against central finite
// differences instead of a hand-derived formula,
TEST(Advanced1DBackwardTest, DeepChainNumericalGradCheck) {
  const std::vector<double> a_val = {1.5, -2.0, 0.5};
  const std::vector<double> b_val = {2.0, 3.0, -1.0};

  auto forward = [&](const std::vector<double>& av,
                     const std::vector<double>& bv) {
    Tape t(TensorClass::TENSOR_ONE, 1.0);
    TensorOne a(av, false, &t);
    TensorOne b(bv, false, &t);
    TensorOne x1 = a + b;
    TensorOne x2 = x1 * a;
    TensorOne x3 = x2 - b;
    TensorOne x4 = x3 * x1;
    TensorOne x5 = x4 + a;
    return x5.data;
  };

  Tape tape(TensorClass::TENSOR_ONE, 1.0);
  TensorOne a(a_val, true, &tape);
  TensorOne b(b_val, true, &tape);
  TensorOne x1 = a + b;
  TensorOne x2 = x1 * a;
  TensorOne x3 = x2 - b;
  TensorOne x4 = x3 * x1;
  TensorOne x5 = x4 + a;

  tape.backward_one(static_cast<int>(x5.node_id));
  const std::vector<double> analytic_da = tape.grad_one[a.node_id];

  constexpr double eps = 1e-6;
  for (size_t k = 0; k < a_val.size(); ++k) {
    std::vector<double> a_plus = a_val, a_minus = a_val;
    a_plus[k] += eps;
    a_minus[k] -= eps;
    const double numeric =
        (forward(a_plus, b_val)[k] - forward(a_minus, b_val)[k]) / (2 * eps);
    EXPECT_NEAR(analytic_da[k], numeric, 1e-4);
  }
}

}  // namespace autograd_tests