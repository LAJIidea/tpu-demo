//
// Created by 王奥博 on 2023/8/29.
//
#include "tpu_mlir/Dialect/Top/Transforms/Passes.h"

#include <regex>

using namespace llvm;

namespace tpu_mlir {
    namespace top {
        class ImportCalibrationTablePass : public ImportCalibrationTableBase<ImportCalibrationTablePass> {
        public:
            ImportCalibrationTablePass() {}
            void runOnOperation() override {

            }
        };

        std::unique_ptr<OperationPass<ModuleOp>> createImportCalibrationTablePass() {
            return std::make_unique<ImportCalibrationTablePass>();
        }
    } // namespace top
} // namespace tpu_mlir