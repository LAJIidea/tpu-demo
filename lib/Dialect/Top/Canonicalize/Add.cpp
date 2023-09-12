//
// Created by kingkiller on 2023/9/5.
//
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"

using namespace tpu_mlir::top;
using namespace tpu_mlir::trait;

struct SwapInput : public OpRewritePattern<AddOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(AddOp op, PatternRewriter &rewriter) const override {
        return failure();
    }
};

struct AddToScale : public OpRewritePattern<AddOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(AddOp op, PatternRewriter &rewriter) const override {
        return failure();
    }
};

struct AddToAddConst : public OpRewritePattern<AddOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(AddOp op, PatternRewriter &rewriter) const override {
        return failure();
    }
};

// Add weight + Add weight
struct AddMerge : public OpRewritePattern<AddOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(AddOp op, PatternRewriter &rewriter) const override {
        return failure();
    }
};

void AddOp::getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context) {
    results.insert<SwapInput, AddToAddConst, AddToScale, AddMerge>(context);
}