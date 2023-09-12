//
// Created by kingkiller on 2023/9/5.
//
#include "tpu_mlir/Support/Module.h"

using namespace tpu_mlir::top;
using namespace tpu_mlir::trait;

// A --slice--> A0 | A1 --concat--> A1 | A0
// ==> SwapDimInner
// test by `test_onnx.py --case SwapDimInner`
struct ConcatToSwapDimInner : public OpRewritePattern<ConcatOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(ConcatOp op, PatternRewriter &rewriter) const override {
        return failure();
    }
};

// concat slices to Depth2Space.
// test by yolov5s
struct ConcatToDepth2SpacePattern : public OpRewritePattern<ConcatOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(ConcatOp op, PatternRewriter &rewriter) const override {
        return failure();
    }
};

struct ConcatToDepth2SpacePattern2 : public OpRewritePattern<ConcatOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(ConcatOp op, PatternRewriter &rewriter) const override {
        return failure();
    }
};

/**
 *       -- Slice --
 *      /           \
 * Op1->|            |->Concat->Op2 => Op1->Slice->Op2
 *      \           /
 *       -- Slice --
 **/
struct MergeSliceConcatPattern : public OpRewritePattern<ConcatOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(ConcatOp op, PatternRewriter &rewriter) const override {
        return failure();
    }
};

struct ConvertLoadWeightConcatToadWeightPattern : public OpRewritePattern<ConcatOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(ConcatOp op, PatternRewriter &rewriter) const override {
        return failure();
    }
};

void ConcatOp::getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context) {
    results.insert<ConvertLoadWeightConcatToadWeightPattern,
                    ConcatToDepth2SpacePattern, ConcatToDepth2SpacePattern2,
                    MergeSliceConcatPattern, ConcatToSwapDimInner>(context);
}