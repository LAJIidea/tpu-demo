//
// Created by 王奥博 on 2023/8/29.
//
#include "tpu_mlir/Dialect/Top/Transforms/Passes.h"
#include "tpu_mlir/Support/Module.h"

using namespace llvm;

namespace tpu_mlir {
    namespace top {

        class InitPass : public InitBase<InitPass> {
        public:
            InitPass() {}
            void runOnOperation() override {

            }
        };


        std::unique_ptr<OperationPass<ModuleOp>> createInitPass() {
            return std::make_unique<InitPass>();
        }
    } // namespace top
} // namespace tpu_mlir