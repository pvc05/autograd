//
// Created by Pranav C on 29/12/2025.
//

#include "tensor_one.h"

namespace autograd {
// TensorBase constructors and method defs

TensorOne::TensorOne(const std::vector<double>& data, const bool requires_grad,
                     Tape* tape)
    : TensorBase(requires_grad, tape, /*node_id=*/-1), data(data) {
  tape->add_node_one(this->data, requires_grad);
}

TensorOne::TensorOne(const int size, const double init_value,
                     const bool requires_grad, Tape* tape)
    : TensorBase(requires_grad, tape, /*node_id=*/-1) {
  this->data = std::vector(size, init_value);
  tape->add_node_one(this->data, requires_grad);
}

size_t TensorOne::size() const { return this->data.size(); }

std::vector<double> TensorOne::value() const { return this->data; }

std::vector<double> TensorOne::grad() const {
  return tape->grad_one[this->node_id];
}

std::string TensorOne::to_string() const {
  std::string repr = "TensorOne(data=[";
  for (size_t i = 0; i < data.size(); ++i) {
    repr += std::to_string(data[i]);
    if (i != data.size() - 1) {
      repr += ", ";
    }
  }
  repr +=
      "], requires_grad=" + std::string(requires_grad ? "true" : "false") + ")";
  return repr;
}

}  // namespace autograd
