//
// Created by kingkiller on 2023/9/5.
//
#include "tpu_mlir/Support/Module.h"

using namespace tpu_mlir::top;

struct RemoveMulConst : public OpRewritePattern<MulConstOp> {
    using OpRewritePattern::OpRewritePattern;
    RemoveMulConst(MLIRContext *context)
            : OpRewritePattern<MulConstOp>(context) {}
    LogicalResult matchAndRewrite(MulConstOp op, PatternRewriter &rewriter) const override {
        return failure();
    }
};

// merge into conv or matmul
struct MergeMulConst : public OpRewritePattern<MulConstOp> {
    using OpRewritePattern::OpRewritePattern;
    MergeMulConst(MLIRContext *context) : OpRewritePattern<MulConstOp>(context) {}
    LogicalResult matchAndRewrite(MulConstOp op, PatternRewriter &rewriter) const override {
        return failure();
    }
};

// mul to large, to 10k
struct MulTooLarge : public OpRewritePattern<MulConstOp> {
    using OpRewritePattern::OpRewritePattern;
    MulTooLarge(MLIRContext *context) : OpRewritePattern<MulConstOp>(context) {}

    LogicalResult matchAndRewrite(MulConstOp op, PatternRewriter &rewriter) const override {
        return failure();
    }
};

void MulConstOp::getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context) {
    results.insert<RemoveMulConst, MergeMulConst, MulTooLarge>(context);
}