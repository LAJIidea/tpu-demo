//
// Created by kingkiller on 2023/9/5.
//
#include "tpu_mlir/Support/Module.h"

using namespace tpu_mlir::top;

// MatMul + Add(weight) => MatMul
struct MatMulWithBias : public OpRewritePattern<MatMulOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(MatMulOp op, PatternRewriter &rewriter) const override {
        return failure();
    }
};

// merge n and c if c is small and n is large
struct OptMatMulSmallCidm : public OpRewritePattern<MatMulOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(MatMulOp op, PatternRewriter &rewriter) const override {
        return failure();
    }
};

// Add Reshape op after non-keepdims MatMul to make layergroup easier
struct NoKeepDimsAddReshape : public OpRewritePattern<MatMulOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(MatMulOp op, PatternRewriter &rewriter) const override {
        return failure();
    }
};

// Matmul + Reshape + Permute0 + (Permute1) + (Reshape2) + n*(slice + squeeze)
// => Matmul + Reshape + n*(slice + squeeze + Permute2 + (Reshape3))
struct MatMulWithPermuteAndSplit : public OpRewritePattern<MatMulOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(MatMulOp op, PatternRewriter &rewriter) const override {
        return failure();
    }
};

void MatMulOp::getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context) {
    results.insert<MatMulWithBias, NoKeepDimsAddReshape, MatMulWithPermuteAndSplit>(context);
}