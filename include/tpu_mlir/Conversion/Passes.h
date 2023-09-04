//
// Created by 王奥博 on 2023/8/29.
//

#pragma once

#include "tpu_mlir/Conversion/Conversion.h"

namespace mlir {
#define GEN_PASS_REGISTRATION
#include "tpu_mlir/Conversion/Passes.h.inc"
} // namespace mlir
