//
// Created by kingkiller on 2023/9/5.
//
#include "tpu_mlir/Support/Module.h"

using namespace tpu_mlir::top;

struct DivToMul : public OpRewritePattern<DivOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(DivOp op, PatternRewriter &rewriter) const override {
        return failure();
    }
};

struct DivToSoftSign : public OpRewritePattern<DivOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(DivOp op, PatternRewriter &rewriter) const override {
        return failure();
    }
};

void DivOp::getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context) {
    results.insert<DivToSoftSign, DivToMul>(context);
}