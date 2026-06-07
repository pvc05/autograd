//
// Created by Pranav C on 29/12/2025.
//

#ifndef AUTOGRAD_BASE_TENSOR_H
#define AUTOGRAD_BASE_TENSOR_H
#pragma once
#include <string>

#include "tape.h"

namespace autograd {

// Base struct to hold common attributes for all tensor types, again will expand
// later to higher dims

struct Tape;

class TensorBase {
 public:
  bool requires_grad;  // does this tensor actually contribute to gradient
  Tape* tape;          // pointer to the tape for comp graph
  int node_id;         // id for comp graph

  explicit TensorBase(bool requires_grad = false, Tape* tape = nullptr,
                      int node_id = -1)
      : requires_grad(requires_grad), tape(tape), node_id(node_id) {};

  void backward();  // initiate backprop from this tensor

  [[nodiscard]] size_t size() const;  // returns the size of the tensor

  [[nodiscard]] std::string to_string() const;

 private:
  size_t id;
};
}  // namespace autograd

#endif  // AUTOGRAD_BASE_TENSOR_H