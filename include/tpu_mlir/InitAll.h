//
// Created by 王奥博 on 2023/8/29.
//

#pragma once

#include "mlir/IR/Dialect.h"

namespace tpu_mlir {
    void registerAllDialects(mlir::DialectRegistry &registry);
    void registerAllPasses();
} // namespace tpu_mlir
