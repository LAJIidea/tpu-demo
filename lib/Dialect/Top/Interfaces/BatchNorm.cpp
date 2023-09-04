//
// Created by 王奥博 on 2023/8/29.
//

#include "tpu_mlir/Support/Module.h"

int64_t top::BatchNormOp::getFLOPs() {
    return module::getNumElements(getOutput()) * 2;
}

LogicalResult top::BatchNormOp::init(InferenceParameter &param) {
    return success();
}

void top::BatchNormOp::deinit(InferenceParameter &param) {}

LogicalResult top::BatchNormOp::inference(InferenceParameter &param) {
    llvm_unreachable("Not Implemented");
    return success();
}

void top::BatchNormOp::shape_inference() {
    common_shape_inference(getOperation());
}