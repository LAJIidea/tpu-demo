//
// Created by 王奥博 on 2023/8/29.
//

#include "tpu_mlir/Support/Module.h"

using namespace tpu_mlir::top;
using namespace tpu_mlir::trait;

struct TopBatchNormToScale : public OpRewritePattern<BatchNormOp> {
    using OpRewritePattern::OpRewritePattern;
    TopBatchNormToScale(MLIRContext *context, PatternBenefit benefit = 9)
        : OpRewritePattern<BatchNormOp>(context, benefit) {}


    LogicalResult matchAndRewrite(BatchNormOp op, PatternRewriter &rewriter) const override {
        return failure();
    }
};

void BatchNormOp::getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context) {
    results.insert<TopBatchNormToScale>(context);
}