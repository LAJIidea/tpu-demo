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
        // 判断是否有其他操作调研了这个输入，如果不止Relu操作使用，则进行算子融合可能会影响其他操作，则停止
        if (!formerOp->getResult(0).hasOneUse())
            return failure();

        // 判断前一个的操作是否含有SupportFuseRelu，这个特性，这个是由tpu-mlir定义的用于判断后续的Relu操作是否运行融入当前操作
        if (!formerOp->hasTrait<SupportFuseRelu>())
            return failure();

        // 这里获取Relu操作的limit限制，小于这个值的一般会进行置为0操作，在我们的td定义中，relu_limit默认值为-1.0
        // 这里的convertToDouble是MLIR内置API，用于将LLVM的内置类型转换为双精度浮点型double
        auto relu_limit = op.getReluLimit().convertToDouble();
        // 这里进一步判断前置操作是否具有relu_limit这个属性，有则会与当前Relu操作限制值进行比较，选择较小的那一个，这确保了融合之后限制值
        // 不会增大
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