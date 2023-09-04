//
// Created by 王奥博 on 2023/8/29.
//

#include "tpu_mlir/Dialect/Top/Transforms/Passes.h"
#include "tpu_mlir/Conversion/Passes.h"
#include "mlir/Transforms/Passes.h"

namespace tpu_mlir {
    void registerAllDialects(mlir::DialectRegistry &registry) {
        registry
            .insert<mlir::tosa::TosaDialect, mlir::func::FuncDialect, top::TopDialect,
                    mlir::quant::QuantizationDialect>();
    }

    void registerAllPasses() {
        registerCanonicalizer();
//        mlir::registerConversionPasses();
        top::registerTopPasses();
    }
}