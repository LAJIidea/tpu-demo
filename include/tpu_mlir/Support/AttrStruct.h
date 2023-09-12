//
// Created by 王奥博 on 2023/8/28.
//

#pragma once
#include <stdint.h>

namespace tpu_mlir {

    typedef struct {
        int64_t n;
        int64_t ic;
        int64_t id;
        int64_t ih;
        int64_t iw;
        int64_t oc;
        int64_t od;
        int64_t oh;
        int64_t ow;
        int64_t kd, dd, sd, ins_d;
        int64_t kh, dh, sh, ins_h;
        int64_t kw, dw, sw, ins_w;
        int64_t pdf, pdb;
        int64_t pht, phb;
        int64_t pwl, pwr;
        int64_t groups;
        int64_t pad_value;
        int64_t kernel_zp;
        int64_t dims; // 1d/2d/3d
        bool has_bias;
        bool is_dw;
        bool do_relu;
        double relu_limit;
    } conv_attr_t;

    typedef struct {
        int64_t batch;
        int64_t M;
        int64_t K;
        int64_t N;
        int64_t batch_low;
        bool with_bias;
        bool do_relu;
        double relu_limit;
        bool left_transpose;
        bool right_transpose;
        bool output_transpose;
        bool hdim_is_batch;
        int64_t input_zp;
        int64_t right_zp;
        int64_t left_reuse;
    } matmul_attr_t;

    typedef struct {
        std::vector<int64_t> in_shape_fix;
        std::vector<int64_t> out_shape_fix;
        std::vector<int64_t> order_fix;
    } permute_attr_t;

} // namespace tpu_mlir
