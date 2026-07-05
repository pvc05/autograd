//
// Created by Pranav C on 29/12/2025.
//

#include "../include/operations.h"

#include <stdexcept>

namespace autograd {

TensorOne operator+(const TensorOne& a, const TensorOne& b) {
  if (a.tape != b.tape) {
    throw std::invalid_argument("Tensors must have the same tape");
  }
  if (a.tape == nullptr) {
    std::vector<double> res(a.size());
    for (size_t i = 0; i < a.size(); ++i) res[i] = a.data[i] + b.data[i];
    return TensorOne(res, false, nullptr);
  }
  const size_t new_id = a.tape->add_one(a.node_id, b.node_id);
  return TensorOne(a.tape, new_id);
}

TensorOne operator-(const TensorOne& a, const TensorOne& b) {
  if (a.tape != b.tape) {
    throw std::invalid_argument("Tensors must have the same tape");
  }
  if (a.tape == nullptr) {
    std::vector<double> res(a.size());
    for (size_t i = 0; i < a.size(); ++i) res[i] = a.data[i] - b.data[i];
    return TensorOne(res, false, nullptr);
  }
  const size_t new_id = a.tape->sub_one(a.node_id, b.node_id);
  return TensorOne(a.tape, new_id);
}

TensorOne operator*(const TensorOne& a, const TensorOne& b) {
  if (a.tape != b.tape) {
    throw std::invalid_argument("Tensors must have the same tape");
  }
  if (a.tape == nullptr) {
    std::vector<double> res(a.size());
    for (size_t i = 0; i < a.size(); ++i) res[i] = a.data[i] * b.data[i];
    return TensorOne(res, false, nullptr);
  }
  const size_t new_id = a.tape->mul_one(a.node_id, b.node_id);
  return TensorOne(a.tape, new_id);
}

}  // namespace autograd
