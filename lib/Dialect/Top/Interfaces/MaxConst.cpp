//
// Created by kingkiller on 2023/9/5.
//
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"

int64_t top::MaxConstOp::getFLOPs() {
    return module::getNumElements(getOutput());
}

LogicalResult top::MaxConstOp::init(InferenceParameter &p) { return success(); }
void top::MaxConstOp::deinit(InferenceParameter &p) {}

LogicalResult top::MaxConstOp::inference(InferenceParameter &p) {
    const int64_t num_elem = module::getNumElements(getOutput());
    const float const_val_ = getConstVal().convertToDouble();
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
    for (int64_t i = 0; i < num_elem; i++) {
        p.outputs[0][i] = std::max(p.inputs[0][i], const_val_);
    }
    return success();
}

void top::MaxConstOp::shape_inference() {
    common_shape_inference(getOperation());
}