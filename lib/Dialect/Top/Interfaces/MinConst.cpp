//
// Created by kingkiller on 2023/9/5.
//
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"

int64_t top::MinConstOp::getFLOPs() {
    return module::getNumElements(getOutput());
}

LogicalResult top::MinConstOp::init(InferenceParameter &p) { return success(); }
void top::MinConstOp::deinit(InferenceParameter &p) {}

LogicalResult top::MinConstOp::inference(InferenceParameter &p) {
    const int64_t num_elem = module::getNumElements(getOutput());
    const float const_val_ = getConstVal().convertToDouble();
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
    for (int64_t i = 0; i < num_elem; i++) {
        p.outputs[0][i] = std::min(p.inputs[0][i], const_val_);
    }
    return success();
}

void top::MinConstOp::shape_inference() {
    common_shape_inference(getOperation());
}