//
// Created by kingkiller on 2023/9/5.
//
#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir::tpu;

#include "tpu_mlir/Dialect/Tpu/IR/TpuOpsDialect.cpp.inc"

void TpuDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.cpp.inc"
    >();
    addAttributes<
#define GET_ATTRDEF_LIST
#include "tpu_mlir/Dialect/Tpu/IR/TpuAttr.cpp.inc"
    >();
}

#include "tpu_mlir/Dialect/Tpu/IR/TpuEnum.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "tpu_mlir/Dialect/Tpu/IR/TpuAttr.cpp.inc"

#define GET_OP_CLASSES
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.cpp.inc"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"


namespace tpu_mlir {
    namespace tpu {
        const conv_attr_t &getConv2DParam(Conv2DOp &op) {
            return {};
        }

        RunMode getRunMode(mlir::func::FuncOp func) {
            return RunMode::CPU;
        }

        void IfOp::getSuccessorRegions(std::optional<unsigned> index,
                                       ArrayRef<Attribute> operands,
                                       SmallVectorImpl<RegionSuccessor> &regions) {
            // The `then` and the `else` region branch back to the parent operation.
            if (index) {
                regions.push_back(RegionSuccessor(getResults()));
                return;
            }

            // Don't consider the else region if it is empty.
            Region *elseRegion = &this->getElseBranch();
            if (elseRegion->empty())
                elseRegion = nullptr;

            // Otherwise, the successor is dependent on the condition.
            bool condition;
            if (auto condAttr = operands.front().dyn_cast_or_null<IntegerAttr>()) {
                condition = condAttr.getValue().isOne();
            } else {
                // If the condition isn't constant, both regions may be executed.
                regions.push_back(RegionSuccessor(&getThenBranch()));
                // If the else region does not exist, it is not a viable successor.
                if (elseRegion)
                    regions.push_back(RegionSuccessor(elseRegion));
                return;
            }

            // Add the successor regions using the condition.
            regions.push_back(RegionSuccessor(condition ? &getThenBranch() : elseRegion));
        }

        void LoopOp::getSuccessorRegions(std::optional<unsigned> index,
                                         ArrayRef<Attribute> operands,
                                         SmallVectorImpl<RegionSuccessor> &regions) {
            // If the predecessor is the ForOp, branch into the body using the iterator
            // arguments.
            if (!index) {
                regions.push_back(RegionSuccessor(&getBody()));
                return;
            }

            // Otherwise, the loop may branch back to itself or the parent operation.
            assert(*index == 0 && "expected loop region");
            regions.push_back(RegionSuccessor(&getBody()));
            regions.push_back(RegionSuccessor(getVFinalAndScanOutputs()));
        }
    }
}
