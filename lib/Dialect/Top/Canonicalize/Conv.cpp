//
// Created by 王奥博 on 2023/8/29.
//

#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir::top;
using namespace mlir;

struct Conv1dTo2d : public OpRewritePattern<ConvOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(ConvOp op, PatternRewriter &rewriter) const override {
        auto a = 1;
        return failure();
    }
};

struct Conv3dTo2d : public OpRewritePattern<ConvOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(ConvOp op, PatternRewriter &rewriter) const override {

        return failure();
    }
};

struct Conv3dTranspose : public OpRewritePattern<ConvOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(ConvOp op, PatternRewriter &rewriter) const override {
        return failure();
    }
};

void ConvOp::getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context) {
    results.insert<Conv3dTranspose, Conv3dTo2d, Conv1dTo2d>(context);
}