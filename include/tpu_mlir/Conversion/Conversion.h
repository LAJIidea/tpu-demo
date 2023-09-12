//
// Created by 王奥博 on 2023/8/29.
//

#pragma once

#include "tpu_mlir/Conversion/TopToTosa/TopLowering.h"
#include "tpu_mlir/Conversion/TopToTpu/TopLowering.h"

namespace mlir {
#define GEN_PASS_DECL
#include "tpu_mlir/Conversion/Passes.h.inc"
} // namespace mlir

namespace tpu_mlir {

    std::unique_ptr<Pass> createConvertTopToTosa();
    std::unique_ptr<Pass> createConvertTopToTpu();

} // namespace tpu_mlir
