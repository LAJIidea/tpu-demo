//
// Created by 王奥博 on 2023/8/29.
//

#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"

#include <iostream>

using namespace tpu_mlir::top;
using namespace tpu_mlir::trait;
using namespace mlir;

// reshape1+permute+reshape2 -> pixelshuffle
// reshape1:[1x128x64x64] -> [1x32x2x2x64x64]
// permute:[1x32x2x2x64x64] -> [1x32x64x2x64x2]
// reshape2:[1x32x64x2x64x2] -> [1x32x128x128]
// ==>pixelshuffle:[1x128x64x64] -> [1x32x128x128]
struct TopPermuteToPixelShuffle : public OpRewritePattern<PermuteOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(PermuteOp op, PatternRewriter &rewriter) const override {
        auto input = op.getInput().getDefiningOp();
        std::cout << input->getName().getStringRef().str() << std::endl;
        Value in = op.getInput();
        if (in.getType().isa<mlir::UnrankedTensorType>()) {
            std::cout << "in is unrank" << std::endl;
        }
        ArrayRef<int64_t> sh = in.getType().cast<UnrankedTensorType>().getShape();
        std::cout << sh.size() << std::endl;
        auto input_shape = module::getShape(op.getInput());
        // 判断输入张量形状大小是否为6
        if (input_shape.size() != 6) {
            return failure();
        }

        std::vector<int64_t> ps_crd = {0, 1, 4, 2, 5, 3};
        std::vector<int64_t> ps_dcr = {0, 3, 4, 1, 5, 2};
        auto order = module::getI64Array(op.getOrder());
        bool crd = true;
        if (*order == ps_crd) {
            crd = true;
        } else if (*order == ps_dcr) {
            crd = false;
        } else {
            return failure();
        }
        auto reshape_before = dyn_cast_or_null<ReshapeOp>(op.getInput().getDefiningOp());
        if (!reshape_before) {
            return failure();
        }
        auto nextOp = *op.getOutput().user_begin();
        auto reshape_after = dyn_cast_or_null<ReshapeOp>(nextOp);
        if (!reshape_after)
            return failure();
        auto output_shape = module::getShape(reshape_after.getOutput());
        int64_t upscale_factor = input_shape[2];
        int64_t on = input_shape[0];
        int64_t oc = crd ? input_shape[1] : input_shape[3];
        int64_t oh = upscale_factor * input_shape[4];
        int64_t ow = upscale_factor * input_shape[5];
        std::vector<int64_t> o_s = {on, oc, oh, ow};
        if (output_shape.vec() != o_s)
            return failure();
        std::vector<NamedAttribute> attrs;
        attrs.push_back(rewriter.getNamedAttr("is_CRD", rewriter.getBoolAttr(crd)));
        attrs.push_back(
                rewriter.getNamedAttr("is_inversed", rewriter.getBoolAttr(false)));
        attrs.push_back(rewriter.getNamedAttr("block_h", rewriter.getI64IntegerAttr(upscale_factor)));
        attrs.push_back(rewriter.getNamedAttr("block_w", rewriter.getI64IntegerAttr(upscale_factor)));
        attrs.push_back(rewriter.getNamedAttr("in_is_NCHW", rewriter.getBoolAttr(true)));
        attrs.push_back(rewriter.getNamedAttr("out_is_NCHW", rewriter.getBoolAttr(true)));
        attrs.push_back(rewriter.getNamedAttr("swap_cr", rewriter.getBoolAttr(false)));
        rewriter.replaceOpWithNewOp<Depth2SpaceOp>(reshape_after, reshape_after.getResult().getType(),
                                                   ValueRange{reshape_before.getInput()}, attrs);
        rewriter.eraseOp(op);
        rewriter.eraseOp(reshape_before);
        return success();
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