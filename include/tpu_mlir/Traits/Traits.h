//
// Created by 王奥博 on 2023/8/28.
//

#pragma once

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

namespace tpu_mlir {
    namespace trait {

        namespace impl {
            mlir::LogicalResult verifyTpuTypeRestrictTrait(mlir::Operation *op);
            mlir::LogicalResult verifyInOutSameShapeTrait(mlir::Operation *op);
        } // namespace impl

// If a op has this trait, it means that some output(s) is(are) shape tensor(s)
        template <typename ConcreteType>
        class ShapeProducer
                : public ::mlir::OpTrait::TraitBase<ConcreteType, ShapeProducer> {};

// If a op has this trait, it means that some input(s) is(are) shape tensor(s)
        template <typename ConcreteType>
        class ShapeConsumer
                : public ::mlir::OpTrait::TraitBase<ConcreteType, ShapeConsumer> {};

// If a op has this trait, it means that relu follow this op can be fused to
// this op
        template <typename ConcreteType>
        class SupportFuseRelu
                : public ::mlir::OpTrait::TraitBase<ConcreteType, SupportFuseRelu> {};

        template <typename ConcreteType>
        class SupportPermuteMove
                : public ::mlir::OpTrait::TraitBase<ConcreteType, SupportPermuteMove> {};

        template <typename ConcreteType>
        class SupportEarlyStride
                : public ::mlir::OpTrait::TraitBase<ConcreteType, SupportEarlyStride> {};

        template <typename ConcreteType>
        class TpuTypeRestrict
                : public ::mlir::OpTrait::TraitBase<ConcreteType, TpuTypeRestrict> {
        public:
            static mlir::LogicalResult verifyTrait(mlir::Operation *op) {
                return impl::verifyTpuTypeRestrictTrait(op);
            }
        };

        template <typename ConcreteType>
        class InOutSameShape
                : public ::mlir::OpTrait::TraitBase<ConcreteType, InOutSameShape> {
        public:
            static mlir::LogicalResult verifyTrait(mlir::Operation *op) {
                return impl::verifyInOutSameShapeTrait(op);
            }
        };

    } // namespace trait
} // namespace tpu_mlir
