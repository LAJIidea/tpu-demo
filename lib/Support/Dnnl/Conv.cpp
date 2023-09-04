//
// Created by 王奥博 on 2023/8/29.
//

#include "tpu_mlir/Support/Dnnl/Conv.h"
#include "tpu_mlir/Support/Dnnl/DnnlUtils.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace dnnl;
using namespace tpu_mlir;

Conv::Conv() {
    eng = dnnl::engine(engine::kind::cpu, 0);
    eng_stream = dnnl::stream(eng);
    memset(&_attr, 0, sizeof(conv_attr_t));
    backw_init = false;
}

Conv::~Conv() {}

void Conv::setup(float *input, float *weight, float *bias, float *output, conv_attr_t attr) {

}

void Conv::run() {

}

void Conv::run_backw(void *dst_grd_input, void *weight_grd_output) {

}
