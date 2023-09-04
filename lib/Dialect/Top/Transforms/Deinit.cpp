//
// Created by 王奥博 on 2023/8/29.
//
#include "tpu_mlir/Dialect/Top/Transforms/Passes.h"

using namespace llvm;

namespace tpu_mlir {
    namespace top {

        class DeinitPass : public DeinitBase<DeinitPass> {
        public:
            DeinitPass() {}
            void runOnOperation() override {

            }
        };

        std::unique_ptr<OperationPass<ModuleOp>> createDeinitPass() {
            return std::make_unique<DeinitPass>();
        }
    }
}