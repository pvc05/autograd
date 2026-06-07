//
// Created by Pranav C on 12/01/2026.
//
#include <cassert>
#include "tape.h"

namespace autograd {

// Add a TENSOR_ONE constant node to the tape
size_t Tape::add_node_one(std::vector<double> value, const bool requires_grad) {
  assert(tape_class == TensorClass::TENSOR_ONE);
  const size_t id = nodes.size(); // new node id is current size of nodes vector

  nodes.push_back(Node{CONST,{}});

  // store node info
  required_grads.push_back(requires_grad);
  values_one.emplace_back(std::move(value)); // store a copy of value

  grad_one.emplace_back(value.size(), seed_grad);
  return id;
}



// Addition of two TENSOR_ONE tensors
size_t Tape::add_one(size_t a, size_t b) {
  assert(tape_class == TensorClass::TENSOR_ONE);

  const std::vector<double> value_one = values_one[a];
  const std::vector<double> value_two = values_one[b];

  assert(value_one.size() == value_two.size() && "Tensor size mismatch in addition");

  const size_t id = nodes.size(); // new node id is current size of nodes vector

  nodes.push_back(Node{ADD,{a,b}});

  // store node info
  required_grads.push_back(required_grads[a] || required_grads[b]);
  std::vector<double> result(value_one.size());

  for (size_t i = 0; i < value_one.size(); ++i) {
    result[i] = value_one[i] + value_two[i];
  }

  values_one.emplace_back(std::move(result)); // store a copy of value
  grad_one.emplace_back(value_one.size(), seed_grad);
  return id;
}

// Subtraction of two TENSOR_ONE tensors, tensor a - tensor b
size_t Tape::sub_one(size_t a, size_t b) {
  assert(tape_class == TensorClass::TENSOR_ONE);

  const std::vector<double> value_one = values_one[a];
  const std::vector<double> value_two = values_one[b];

  assert(value_one.size() == value_two.size() && "Tensor size mismatch in subtraction");

  const size_t id = nodes.size(); // new node id is current size of nodes vector

  nodes.push_back(Node{SUB,{a,b}});

  // store node info
  required_grads.push_back(required_grads[a] || required_grads[b]);
  std::vector<double> result(value_one.size());

  for (size_t i = 0; i < value_one.size(); ++i) {
    result[i] = value_one[i] - value_two[i];
  }

  values_one.emplace_back(std::move(result)); // store a copy of value
  grad_one.emplace_back(value_one.size(), seed_grad);
  return id;
}

// Multiplication of two TENSOR_ONE tensors
size_t Tape::mul_one(size_t a, size_t b) {
  assert(tape_class == TensorClass::TENSOR_ONE);

  const std::vector<double> value_one = values_one[a];
  const std::vector<double> value_two = values_one[b];

  assert(value_one.size() == value_two.size() && "Tensor size mismatch in multiplication");

  const size_t id = nodes.size(); // new node id is current size of nodes vector

  nodes.push_back(Node{MUL,{a,b}});

  // store node info
  required_grads.push_back(required_grads[a] || required_grads[b]);
  std::vector<double> result(value_one.size());

  for (size_t i = 0; i < value_one.size(); ++i) {
  result[i] = value_one[i] * value_two[i];
  }

  values_one.emplace_back(std::move(result)); // store a copy of value
  grad_one.emplace_back(value_one.size(), seed_grad);
  return id;
}

}  // namespace autograd