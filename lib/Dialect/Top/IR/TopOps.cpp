//
// Created by 王奥博 on 2023/8/28.
//
#include "tpu_mlir/Support/Module.h"

using namespace tpu_mlir::top;

//===---------------------------===//
// Dialect initialize method.
//===---------------------------===//
#include "tpu_mlir/Dialect/Top/IR/TopOpsDialect.cpp.inc"

void TopDialect::initialize() {
    addAttributes<
#define GET_ATTRDEF_LIST
    >();
    addOperations<
#define GET_OP_LIST
#include "tpu_mlir/Dialect/Top/IR/TopOps.cpp.inc"
    >();
}

//===--------------------------===//
// Top Operator Definitions.
//===--------------------------===//
#define GET_ATTRDEF_CLASSES

#define GET_OP_CLASSES
#include "tpu_mlir/Dialect/Top/IR/TopOps.cpp.inc"
