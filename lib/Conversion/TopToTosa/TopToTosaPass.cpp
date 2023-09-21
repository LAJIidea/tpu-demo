//
// Created by 王奥博 on 2023/8/29.
//
#include "tpu_mlir/Conversion/TopToTosa/OpLowering.h"
#include "tpu_mlir/Conversion/Conversion.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTTOPTOTOSA
#include "tpu_mlir/Conversion/Passes.h.inc"
} // namespace mlir

namespace tpu_mlir {

struct LoweringTopWeightOp : public OpRewritePattern<top::WeightOp> {
public:
    LoweringTopWeightOp(MLIRContext *ctx, bool include_weight)
        : OpRewritePattern(ctx), include_weight(include_weight) {}

    LogicalResult matchAndRewrite(top::WeightOp op, PatternRewriter &rewriter) const override {
        assert(op->getNumResults() == 1);
        auto outType = change_dataformat(op->getResult(0).getType());
        auto has_weight = include_weight;
        for (auto user : op.getOutput().getUsers()) {
            if (isa<tosa::TransposeOp>(user)) {
                has_weight = true;
            }
        }
        if (has_weight) {
            auto valptr = op.read_as_float();
            auto new_val = change_weight(valptr, op->getResult(0).getType());
            auto attr =
                    DenseElementsAttr::get(outType.cast<RankedTensorType>(),
                            llvm::ArrayRef(new_val, valptr->size()));
            rewriter.replaceOpWithNewOp<mlir::tosa::ConstOp>(op, outType, attr);
        } else {
            auto attr = DenseElementsAttr::get(
                    RankedTensorType::get({}, rewriter.getI64Type()),
                    llvm::ArrayRef<int64_t>({0})
                    );
            rewriter.replaceOpWithNewOp<mlir::tosa::ConstOp>(op, outType, attr);
        }
        return success();
    }

private:
    bool include_weight;
};

struct EraseTopNoneOp : public OpRewritePattern<top::NoneOp> {
public:
    using OpRewritePattern<top::NoneOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(top::NoneOp op, PatternRewriter &rewriter) const override {
        rewriter.eraseOp(op);
        return success();
    }
};

struct ConvertTopToTosa : public ::impl::ConvertTopToTosaBase<ConvertTopToTosa> {
public:
    void runOnOperation() override {
        module_ = getOperation();
        ctx_ = &getContext();
        mainFunc_ = module::getMainFuncOp(module_);

        RewritePatternSet patterns(ctx_);
        ConversionTarget target(*ctx_);
        target.addLegalDialect<mlir::tosa::TosaDialect, mlir::func::FuncDialect>();

        // Change data format for FuncOp

        // Lower TOP Ops
        patterns.add<LoweringTopWeightOp>(patterns.getContext(), includeWeight);
        populateTopToTosaConversionPatterns(&patterns);
        auto config = GreedyRewriteConfig();
        config.maxIterations = 1;
        applyPatternsAndFoldGreedily(module_, std::move(patterns), config);

        // Erase TOP::NoneOp
        patterns.clear();
        patterns.add<EraseTopNoneOp>(ctx_);
        applyPatternsAndFoldGreedily(module_, std::move(patterns));

        module::updateModuleTypes();
        module::setState(module::State::TOSA_F32);
    }

protected:
    ModuleOp module_;
    FuncOp mainFunc_;
    MLIRContext *ctx_;
};

std::unique_ptr<Pass> createConvertTopToTosa() {
    return std::make_unique<ConvertTopToTosa>();
}
}