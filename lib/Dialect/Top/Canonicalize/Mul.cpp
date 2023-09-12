//
// Created by kingkiller on 2023/9/5.
//
#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir::top;

struct MulToSiLU : public OpRewritePattern<MulOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(MulOp op, PatternRewriter &rewriter) const override {
        return failure();
    }
};

struct MulToMulConst : public OpRewritePattern<MulOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(MulOp op, PatternRewriter &rewriter) const override {
        return failure();
    }
};

struct MulToScale : public OpRewritePattern<MulOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(MulOp op, PatternRewriter &rewriter) const override {
        return failure();
    }
};

// Mul + Mul
struct MulMerge : public OpRewritePattern<MulOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(MulOp op, PatternRewriter &rewriter) const override {
        return failure();
    }
};

struct MergeGelu : public OpRewritePattern<MulOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(MulOp op, PatternRewriter &rewriter) const override {
        return failure();
    }
};

void MulOp::getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context) {
    results.insert<MulToSiLU, MulToMulConst, MulToScale, MulMerge, MergeGelu>(context);
}