//
// Created by 王奥博 on 2023/8/29.
//

#pragma once
#include "oneapi/dnnl/dnnl.hpp"
using namespace dnnl;
namespace tpu_mlir {

    void post_relu(primitive_attr &attr, bool &do_relu, double &relu_limit);
} // namespace tpu_mlir
