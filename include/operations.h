//
// Created by Pranav C on 29/12/2025.
//

#ifndef AUTOGRAD_OPERATIONS_H
#define AUTOGRAD_OPERATIONS_H

#include "tensor_one.h"

namespace autograd {

TensorOne operator+(const TensorOne& a, const TensorOne& b);
TensorOne operator-(const TensorOne& a, const TensorOne& b);
TensorOne operator*(const TensorOne& a, const TensorOne& b);

}  // namespace autograd

#endif  // AUTOGRAD_OPERATIONS_H