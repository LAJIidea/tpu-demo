//
// Created by 王奥博 on 2023/8/29.
//

#include "tpu_mlir/Support/Module.h"

using namespace tpu_mlir::top;
using namespace tpu_mlir::trait;

struct TopFuseRelu : public OpRewritePattern<ReluOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(ReluOp op, PatternRewriter &rewriter) const override {
        auto formerOp = op.getInput().getDefiningOp();
        if (!formerOp->getResult(0).hasOneUse())
            return failure();

        if (!formerOp->hasTrait<SupportFuseRelu>())
            return failure();
        auto relu_limit = op.getReluLimit().convertToDouble();
        if (formerOp->hasAttr("relu_limit")) {
            auto old_limit =
                    formerOp->getAttr("relu_limit").cast<FloatAttr>().getValueAsDouble();
            if (old_limit > 0 && relu_limit > old_limit) {
                relu_limit = old_limit;
            }
        }
        formerOp->setAttr("do_relu", rewriter.getBoolAttr(true));
        formerOp->setAttr("relu_limit", rewriter.getF64FloatAttr(relu_limit));
        formerOp->setLoc(op->getLoc());
        // remove the relu Op
        rewriter.replaceOp(op, {op.getInput()});
        return success();
    }
};

struct TopMoveReluAheadConcatPattern : public OpRewritePattern<ReluOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(ReluOp op, PatternRewriter &rewriter) const override {
        return failure();
    }
};

void ReluOp::getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context) {
    results.insert<TopMoveReluAheadConcatPattern, TopFuseRelu>(context);
}