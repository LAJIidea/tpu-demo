//
// Created by 王奥博 on 2023/9/1.
//
#include "tpu_mlir/Support/Module.h"

using namespace tpu_mlir::top;
using namespace tpu_mlir::trait;

struct TopMultiScaleMergeToOne : public OpRewritePattern<ScaleOp> {
    using OpRewritePattern::OpRewritePattern;

    TopMultiScaleMergeToOne(MLIRContext *context, PatternBenefit benefit = 10)
        : OpRewritePattern(context, benefit) {}

    LogicalResult matchAndRewrite(ScaleOp op, PatternRewriter &rewriter) const override {
        return failure();
    }
};

struct ConstbinaryMergerToTopScale : public OpRewritePattern<ScaleOp> {
    using OpRewritePattern::OpRewritePattern;
    ConstbinaryMergerToTopScale(MLIRContext *context, PatternBenefit benefit = 6)
        : OpRewritePattern<ScaleOp>(context, benefit) {}

    LogicalResult matchAndRewrite(ScaleOp op, PatternRewriter &rewriter) const override {
        return failure();
    }
};

struct TopScaleMergeToBatchNorm : public OpRewritePattern<ScaleOp> {
    using OpRewritePattern::OpRewritePattern;
    TopScaleMergeToBatchNorm(MLIRContext *context, PatternBenefit benefit = 9)
        : OpRewritePattern<ScaleOp>(context, benefit) {}

    LogicalResult matchAndRewrite(ScaleOp op, PatternRewriter &rewriter) const override {
        return failure();
    }
};

struct ScaleShapeAlign : public OpRewritePattern<ScaleOp> {
    using OpRewritePattern::OpRewritePattern;
    ScaleShapeAlign(MLIRContext *context, PatternBenefit benefit = 1)
        : OpRewritePattern<ScaleOp>(context, benefit) {}

    LogicalResult matchAndRewrite(ScaleOp op, PatternRewriter &rewriter) const override {
        return failure();
    }
};

struct TopScaleMergeToMatMul : public OpRewritePattern<ScaleOp> {
    using OpRewritePattern::OpRewritePattern;
    TopScaleMergeToMatMul(MLIRContext *context, PatternBenefit benefit = 1)
        : OpRewritePattern<ScaleOp>(context, benefit) {}

    LogicalResult matchAndRewrite(ScaleOp op, PatternRewriter &rewriter) const override {
        return failure();
    }
};

struct FuseScaleIntConv : public OpRewritePattern<ScaleOp> {
    using OpRewritePattern::OpRewritePattern;
    FuseScaleIntConv(MLIRContext *context, PatternBenefit benefit = 1)
        : OpRewritePattern<ScaleOp>(context, benefit) {}

    LogicalResult matchAndRewrite(ScaleOp op, PatternRewriter &rewriter) const override {
        return failure();
    }
};

void ScaleOp::getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context) {
    results.insert<TopMultiScaleMergeToOne, TopScaleMergeToBatchNorm,
                   ScaleShapeAlign, ConstbinaryMergerToTopScale,
                   TopScaleMergeToMatMul, FuseScaleIntConv>(context);
}