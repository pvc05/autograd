//
// Created by Pranav C on 29/12/2025.
//

#ifndef AUTOGRAD_TENSOR_ONE_H
#define AUTOGRAD_TENSOR_ONE_H
#include <string>
#include <vector>

#include "tensor_base.h"

// tensor_one refers to a 1D tensor, i.e., a vector.
// Might expand this to higher dims in the future once this works, hence the
// base tensor struct.

namespace autograd {

class TensorOne : public TensorBase {
 public:
  std::vector<double> data;  // underlying data, imp note: this makes a copy
  bool requires_grad{};

  // Constructors, one for data vector, one for size + init value vector
  explicit TensorOne(const std::vector<double>& data,
                     bool requires_grad = false, Tape* tape = nullptr);
  explicit TensorOne(size_t size, double init_value = 0.0,
                     bool requires_grad = false, Tape* tape = nullptr);
  explicit TensorOne(Tape* tape, size_t node_id);

  [[nodiscard]] std::vector<double> value() const;
  [[nodiscard]] std::vector<double> grad() const;
  [[nodiscard]] size_t size() const;
  void backward();
  [[nodiscard]] std::string to_string() const;
};

}  // namespace autograd

#endif  // AUTOGRAD_TENSOR_ONE_H