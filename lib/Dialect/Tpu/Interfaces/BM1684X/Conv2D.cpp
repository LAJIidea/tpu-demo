//
// Created by kingkiller on 2023/9/6.
//
#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.h"
#include "tpu_mlir/Support/MathUtils.h"

void tpu::Conv2DOp::codegen_global_bm1684x() {
}

// ======================================
// LocalGenInterface
// ======================================

int64_t tpu::Conv2DOp::getBufferSize_bm1684x(
        int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
        int64_t in_cslice, int64_t in_hslice, int64_t in_dslice, int64_t in_wslice,
        int64_t out_nslice, int64_t out_cslice, int64_t out_hslice,
        int64_t out_dslice, int64_t out_wslice, group_type_t group_type) {

    int64_t sz = 0;
    return sz;
}

void tpu::Conv2DOp::codegen_local_bm1684x(int64_t n_step, int64_t c_step,
                                          int64_t h_step, int64_t d_step,
                                          int64_t w_step,
                                          group_type_t group_type,
                                          local_sec_info_t &sec_info) {

}

// dynamic codegen
int64_t tpu::Conv2DOp::dyn_codegen_local_bm1684x(void *buffer) {
    return 0;
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::Conv2DOp::dyn_codegen_global_bm1684x(void *buffer) {
    return 0;
}

int64_t tpu::Conv2DOp::get_fw_type_bm1684x() { return 0; }
