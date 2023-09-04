//
// Created by 王奥博 on 2023/8/29.
//

#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Support/MathUtils.h"

int64_t top::ReluOp::getFLOPs() {
    return module::getNumElements(getOutput());
}

LogicalResult top::ReluOp::init(InferenceParameter &param) {
    return success();
}

void top::ReluOp::deinit(InferenceParameter &param) {}

LogicalResult top::ReluOp::inference(InferenceParameter &param) {
    auto limit = getReluLimit().convertToDouble();
    function_relu(param.inputs[0], param.outputs[0], module::getNumElements(getInput()),
                 limit);
    return success();
}

void top::ReluOp::shape_inference() {
    common_shape_inference(getOperation());
}