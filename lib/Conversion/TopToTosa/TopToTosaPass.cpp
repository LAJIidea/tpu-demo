//
// Created by 王奥博 on 2023/8/29.
//
#include "tpu_mlir/Conversion/TopToTosa/OpLowering.h"
#include "tpu_mlir/Conversion/Conversion.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTTOPTOTOSA
#include "tpu_mlir/Conversion/Passes.h.inc"
} // namespace mlir

namespace tpu_mlir {
struct ConvertTopToTosa : public ::impl::ConvertTopToTosaBase<ConvertTopToTosa> {
public:
    void runOnOperation() override {

    }
};

std::unique_ptr<Pass> createConvertTopToTosa() {
    return std::make_unique<ConvertTopToTosa>();
}
}