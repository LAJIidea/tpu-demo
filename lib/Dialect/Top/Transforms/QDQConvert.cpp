//
// Created by 王奥博 on 2023/8/29.
//
#include "tpu_mlir/Dialect/Top/Transforms/Passes.h"

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace llvm;

namespace tpu_mlir {
    namespace top {
        class QDQConvertPass : public QDQConvertBase<QDQConvertPass> {
        public:
            QDQConvertPass() {}
            void runOnOperation() override {

            }
        };

        std::unique_ptr<OperationPass<ModuleOp>> createQDQConvertPass() {
            return std::make_unique<QDQConvertPass>();
        }
    } // namespace top
} // namespace tpu_mlir