//
// Created by 王奥博 on 2023/8/28.
//

#pragma once

#include "mlir/IR/OpDefinition.h"

namespace tpu_mlir {
    // only one output, and output shape is the same with the first input shape
    void common_shape_inference(mlir::Operation *op);

    // binary operation with the implicit broadcast
    void broadcast_shape_inference(mlir::Operation *op);

    // reshape broadcast tensor if shape dim is not same with the expected tensor
    void broadcast_tensor_reshape(const mlir::Value &expect, mlir::Value &input);
} // namespace tpu_mlir

#include "tpu_mlir/Interfaces/ShapeInterface.h.inc"