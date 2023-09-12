//
// Created by kingkiller on 2023/9/6.
//
#include "tpu_mlir/Support/MathUtils.h"

// ======================================
// GlobalGenInterface
// ======================================

void tpu::Conv2DOp::codegen_global_cv18xx(int64_t layer_id) {

}

// ======================================
// LocalGenInterface
// ======================================

int64_t tpu::Conv2DOp::getBufferSize_cv18xx(
        int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
        int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {

    return 0;
}

void tpu::Conv2DOp::codegen_local_cv18xx(int64_t n_step, int64_t h_step,
                                         int64_t d_step, int64_t w_step,
                                         group_type_t group_type,
                                         local_sec_info_t &sec_info,
                                         int64_t layer_id) {

}
