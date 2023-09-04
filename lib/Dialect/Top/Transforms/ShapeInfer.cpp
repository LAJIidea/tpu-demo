//
// Created by 王奥博 on 2023/8/29.
//
#include "tpu_mlir/Dialect/Top/Transforms/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"
#define DEBUG_TYPE "shape_infer"

using namespace llvm;

namespace tpu_mlir {
    namespace top {

        class ShapeInferPass : public ShapeInferBase<ShapeInferPass> {
        public:
            ShapeInferPass() {}
            void runOnOperation() override {

            }
        };

        std::unique_ptr<OperationPass<ModuleOp>> createShapeInferPass() {
            return std::make_unique<ShapeInferPass>();
        }
    } // namespace top
} // namespace tpu_mlir
