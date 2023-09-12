//
// Created by kingkiller on 2023/9/5.
//
#include "tpu_mlir/Support/Module.h"

using namespace tpu_mlir::top;

class LoopOpRewriteMaxTripleCountPattern : public OpRewritePattern<LoopOp> {
public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(LoopOp op, PatternRewriter &rewriter) const override {
        return failure();
    }
};

void LoopOp::getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context) {
    results.insert<LoopOpRewriteMaxTripleCountPattern>(context);
}