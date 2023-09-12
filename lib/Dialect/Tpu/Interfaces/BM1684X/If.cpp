//
// Created by kingkiller on 2023/9/6.
//
#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"

// =========================================
// GlobalGenInterface
// =========================================
void tpu::IfOp::codegen_global_bm1684x() {
    llvm_unreachable("Only support dynamic codegen");
}

int64_t tpu::IfOp::dyn_codegen_global_bm1684x(void *buffer) {
    return 0;
}

int64_t tpu::IfOp::get_fw_type_bm1684x() {
    return 0;
}