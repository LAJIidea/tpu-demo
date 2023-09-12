//
// Created by kingkiller on 2023/9/6.
//
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"

LogicalResult tpu::IfOp::init(InferenceParameter &p) {
    return success();
}

void tpu::IfOp::deinit(InferenceParameter &p) {
}

LogicalResult tpu::IfOp::inference(InferenceParameter &p) {
    if (p.inputs[0][0] > 0)
        return success(); //then_branch
    else
        return failure(); //else_branch
}