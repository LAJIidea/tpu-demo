//
// Created by kingkiller on 2023/9/5.
//
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir::top;

struct MaxToMaxConst : public OpRewritePattern<MaxOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(MaxOp op, PatternRewriter &rewriter) const override {
        return failure();
    }
};

void MaxOp::getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context) {
    results.insert<MaxToMaxConst>(context);
}