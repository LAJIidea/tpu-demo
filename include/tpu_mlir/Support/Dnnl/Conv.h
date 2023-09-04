//
// Created by 王奥博 on 2023/8/29.
//

#pragma once
#include "oneapi/dnnl/dnnl.hpp"
#include "tpu_mlir/Support/AttrStruct.h"

using namespace dnnl;

namespace tpu_mlir {

    class Conv {
    public:
        Conv();
        ~Conv();

        void setup(float *input, float *weight, float *bias, float *output,
                   conv_attr_t attr);
        void run();
        void run_backw(void *dst_grd_input, void *weight_grd_output);

    private:
        engine eng;
        stream eng_stream;
        conv_attr_t _attr;

        bool backw_init;
    };
} // namespace tpu_mlir
