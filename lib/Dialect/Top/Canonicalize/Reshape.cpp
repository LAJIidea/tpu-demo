//
// Created by 王奥博 on 2023/8/29.
//

#include "tpu_mlir/Support/Module.h"

using namespace tpu_mlir::top;

struct TopFuseReshape2 : public OpRewritePattern<ReshapeOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(ReshapeOp op, PatternRewriter &rewriter) const override {
        Value output = op.getOutput();
        auto shape0 = module::getShape(output);
        auto shape1 = module::getShape(op.getInput());
        if (shape0 != shape1) {
            return failure();
        }
        op.getOutput().replaceAllUsesWith(op.getInput());
        rewriter.eraseOp(op);
        return success();
    }
};

struct TopFuseReshape3 : public OpRewritePattern<ReshapeOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(ReshapeOp op, PatternRewriter &rewriter) const override {
        return failure();
    }
};

struct ReshapeInstanceNormPattern : public OpRewritePattern<ReshapeOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(ReshapeOp op, PatternRewriter &rewriter) const override {
        return failure();
    }
};

struct MergeGeluPattern : public OpRewritePattern<ReshapeOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(ReshapeOp op, PatternRewriter &rewriter) const override {
        return failure();
    }
};

void ReshapeOp::getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context) {
    results.insert<TopFuseReshape2,
                   TopFuseReshape3,
                   ReshapeInstanceNormPattern,
                   MergeGeluPattern>(context);
}