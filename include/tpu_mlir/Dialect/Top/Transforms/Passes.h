//
// Created by 王奥博 on 2023/8/29.
//

#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/QuantOps.h"
#include "tpu_mlir/Dialect/Top/IR/TopOps.h"

using namespace mlir;

namespace tpu_mlir {
    namespace top {
        std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createInitPass();
        std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createDeinitPass();
        std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createImportCalibrationTablePass();
        std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createQDQConvertPass();
        std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createExtraOptimizePass();
        std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createShapeInferPass();

#define GEN_PASS_REGISTRATION
#define GEN_PASS_CLASSES
#include "tpu_mlir/Dialect/Top/Transforms/Passes.h.inc"
    } // namespace top
} // namespace tpu_mlir