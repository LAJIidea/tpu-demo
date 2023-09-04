//
// Created by 王奥博 on 2023/8/29.
//

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tpu_mlir/Dialect/Top/Transforms/Passes.h"

using namespace llvm;

namespace tpu_mlir {
    namespace top {
        class ExtraOptimizePass : public ExtraOptimizeBase<ExtraOptimizePass> {
        public:
            ExtraOptimizePass() {}
            void runOnOperation() override {

            }
        };

        std::unique_ptr<OperationPass<ModuleOp>> createExtraOptimizePass() {
            return std::make_unique<ExtraOptimizePass>();
        }
    } // namespace top
} //namespace tpu_mlir