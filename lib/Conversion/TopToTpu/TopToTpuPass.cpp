//
// Created by kingkiller on 2023/9/5.
//
#include "tpu_mlir/Conversion/Conversion.h"
#include <regex>

namespace mlir {
#define GEN_PASS_DEF_CONVERTTOPTOTPU
#include "tpu_mlir/Conversion/Passes.h.inc"
} // namespace mlir

namespace tpu_mlir {
struct ConvertTopToTpu : public ::impl::ConvertTopToTpuBase<ConvertTopToTpu> {
public:
    void runOnOperation() override {

    }
};

std::unique_ptr<Pass> createConvertTopToTpu() {
    return std::make_unique<ConvertTopToTpu>();
}
}