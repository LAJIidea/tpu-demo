//
// Created by kingkiller on 2023/9/5.
//
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir::top;

struct MinToMinConst : public OpRewritePattern<MinOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(MinOp op, PatternRewriter &rewriter) const override {
        return failure();
    }
};

void MinOp::getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context) {
    results.insert<MinToMinConst>(context);
}