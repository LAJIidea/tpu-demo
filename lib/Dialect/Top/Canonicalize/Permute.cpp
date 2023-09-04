//
// Created by 王奥博 on 2023/8/29.
//

#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir::top;
using namespace tpu_mlir::trait;
using namespace mlir;

struct TopPermuteToPixelShuffle : public OpRewritePattern<PermuteOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(PermuteOp op, PatternRewriter &rewriter) const override {
        return failure();
    }
};

struct TopPermuteToReorg : public OpRewritePattern<PermuteOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(PermuteOp op, PatternRewriter &rewriter) const override {
        return failure();
    }
};

struct Permute5dSplit : public OpRewritePattern<PermuteOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(PermuteOp op, PatternRewriter &rewriter) const override {
        return failure();
    }
};

struct PermuteFuse : public OpRewritePattern<PermuteOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(PermuteOp op, PatternRewriter &rewriter) const override {
        return failure();
    }
};

struct TopPermuteToReshape : public OpRewritePattern<PermuteOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(PermuteOp op, PatternRewriter &rewriter) const override {
        return failure();
    }
};

struct SoftmaxPermutePattern : public OpRewritePattern<PermuteOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(PermuteOp op, PatternRewriter &rewriter) const override {
        return failure();
    }
};

/**
 * Op1->NonZero->Permute->Op2 => Op1->NonZero->Op2
 **/
struct NonZeroPermutePattern : public OpRewritePattern<PermuteOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(PermuteOp op, PatternRewriter &rewriter) const override {
        return failure();
    }
};

// permute + pad -> pad + permute
struct PermutePadSwap : public OpRewritePattern<PermuteOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(PermuteOp op, PatternRewriter &rewriter) const override {
        return failure();
    }
};

/**
 * Op1 -> perm -> next  => Op1 -> next -> perm
 **/
struct PermuteMovePattern : public OpRewritePattern<PermuteOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(PermuteOp op, PatternRewriter &rewriter) const override {
        return failure();
    }
};

/**
 * Permute(x2)->Binary => Binary->Permute
 **/
struct PermuteBinaryPattern : public OpRewritePattern<PermuteOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(PermuteOp op, PatternRewriter &rewriter) const override {
        return failure();
    }
};

void PermuteOp::getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context) {
    results.insert<TopPermuteToPixelShuffle, TopPermuteToReorg, Permute5dSplit,
                   PermuteFuse, PermuteMovePattern, TopPermuteToReshape,
                   SoftmaxPermutePattern, NonZeroPermutePattern, PermutePadSwap,
                   PermuteBinaryPattern>(context);
}