//
// Created by 王奥博 on 2023/8/29.
//

#include "tpu_mlir/Support/Module.h"

#include <iostream>

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

// add + reshape + add + reshape
struct TopFuseReshape3 : public OpRewritePattern<ReshapeOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(ReshapeOp op, PatternRewriter &rewriter) const override {
//        auto in = op.getInput();
        return failure();
    }
};

// reshape<(0, ng, -1)> + instance_norm -> group_norm<ng> + reshape
struct ReshapeInstanceNormPattern : public OpRewritePattern<ReshapeOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(ReshapeOp op, PatternRewriter &rewriter) const override {

        return failure();
    }
};

// merge some tanh and power(x, 3) comprised gelu to gelu, first found in pytorch traced gpt2
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