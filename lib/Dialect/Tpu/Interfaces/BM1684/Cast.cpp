//
// Created by kingkiller on 2023/9/6.
//
#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"

void tpu::CastOp::codegen_global_bm1684() {
    int64_t n, c, h, w;
    module::getNCHW(getOutput(), n, c, h, w);

}

int64_t tpu::CastOp::getBufferSize_bm1684(int64_t in_lmem_bytes,
                                          int64_t out_lmem_bytes,
                                          int64_t in_nslice, int64_t in_hslice,
                                          int64_t out_nslice,
                                          int64_t out_hslice) {
    int64_t local_buffer_size = 0;
    return local_buffer_size;
}

void tpu::CastOp::codegen_local_bm1684(int64_t n_step, int64_t h_step,
                                       local_sec_info_t &sec_info) {
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================

uint32_t tpu::CastOp::dyn_codegen_global_bm1684(void *ir_layer_info) {
    uint32_t fw_ir_length = 0;
    return fw_ir_length;
}

int64_t tpu::CastOp::get_fw_type_bm1684() { return 0; }

// ======================================
// Dynamic LocalGenInterface
// ======================================

int32_t tpu::CastOp::dyn_codegen_local_bm1684(void *ir_layer_info) {
    int fw_ir_length = 0;
    // compute fw ir info length for input and output
    fw_ir_length +=
            2 * sizeof(uint32_t) + 2 * sizeof(uint32_t) + sizeof(uint32_t);

    // add fw ir length for output consumer number
    fw_ir_length += sizeof(uint32_t);

    return fw_ir_length;
}
