//
// Created by kingkiller on 2023/9/6.
//
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.h"

// =========================================
// GlobalGenInterface
// =========================================
void tpu::LoopOp::codegen_global_bm1684x() {
    llvm_unreachable("Only support dynamic codegen");
}

int64_t tpu::LoopOp::dyn_codegen_global_bm1684x(void *buffer) {
    return 0;
}

int64_t tpu::LoopOp::get_fw_type_bm1684x() {
    return 0;
}