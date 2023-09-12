//
// Created by kingkiller on 2023/9/5.
//
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"

int64_t top::AddConstOp::getFLOPs() {
    return module::getNumElements(getOutput()) * (1 + (getDoRelu() ? 1 : 0));
}

LogicalResult top::AddConstOp::init(InferenceParameter &p) { return success(); }
void top::AddConstOp::deinit(InferenceParameter &p) {}

LogicalResult top::AddConstOp::inference(InferenceParameter &p) {
    const int64_t num_elem = module::getNumElements(getOutput());
    const float const_val_ = getConstVal().convertToDouble();
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
    for (int64_t i = 0; i < num_elem; i++) {
        p.outputs[0][i] = p.inputs[0][i] + const_val_;
    }
    if (getDoRelu()) {
        auto limit = getReluLimit().convertToDouble();
        function_relu(p.outputs[0], p.outputs[0], num_elem, limit);
    }
    return success();
}

void top::AddConstOp::shape_inference() {
    common_shape_inference(getOperation());
}