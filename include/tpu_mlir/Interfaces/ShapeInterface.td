#ifndef TPU_MLIR_SHAPEINTERFACE
#define TPU_MLIR_SHAPEINTERFACE

include "mlir/IR/OpBase.td"

def ShapeInterface : OpInterface<"ShapeInterface"> {
    let description = [{
    }];
    let cppNamespace = "::tpu_mlir";
    let methods = [
        InterfaceMethod<
                [{
                    if output shape is dynamic, inference output shape by input shape;
                    or verify output shape
            }],
            "void",
            "shape_inference",
            (ins)
        >,
    ];
}
#endif // TPU_MLIR_SHAPEINTERFACE