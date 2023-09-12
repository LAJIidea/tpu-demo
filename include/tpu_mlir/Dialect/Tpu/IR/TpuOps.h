//
// Created by kingkiller on 2023/9/5.
//

#pragma once

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "tpu_mlir/Interfaces/InferenceInterface.h"
#include "tpu_mlir/Interfaces/GlobalGenInterface.h"
#include "tpu_mlir/Interfaces/LocalGenInterface.h"
#include "tpu_mlir/Interfaces/TypeInterface.h"
#include "tpu_mlir/Interfaces/IndexingMapsInterface.h"
#include "tpu_mlir/Support/AttrStruct.h"
#include "tpu_mlir/Traits/Traits.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOpsDialect.h.inc"

#include "tpu_mlir/Dialect/Tpu/IR/TpuEnum.h.inc"
#define GET_ATTRDEF_CLASSES
#include "tpu_mlir/Dialect/Tpu/IR/TpuAttr.h.inc"
#define GET_OP_CLASSES
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h.inc"

namespace tpu_mlir {
    namespace tpu {
        const conv_attr_t &getConv2DParam(tpu::Conv2DOp &op);

        RunMode getRunMode(mlir::func::FuncOp func);
//        RunMode getRunMode(Operation *op);
    } // namespace tpu
} // namespace tpu_mlir
