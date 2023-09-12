//
// Created by kingkiller on 2023/9/6.
//
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"

Value tpu::BufferOp::create(mlir::Operation *OwnerOp, mlir::RankedTensorType &type) {
    OpBuilder builder(OwnerOp->getContext());
    builder.setInsertionPoint(OwnerOp);
    auto loc = module::getLocLike(OwnerOp, "buffer");
    auto newOp = builder.create<tpu::BufferOp>(loc, type);
    return newOp.getResult();
}