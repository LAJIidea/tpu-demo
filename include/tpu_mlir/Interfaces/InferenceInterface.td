#ifndef TPU_MLIR_INFERENCEINTERFACE
#define TPU_MLIR_INFERENCEINTERFACE

include "mlir/IR/OpBase.td"

def InferenceInterface : OpInterface<"InferenceInterface"> {
    let description = [{
    }];
    let cppNamespace = "::tpu_mlir";
    let methods = [
        InterfaceMethod<
            /*desc=*/[{
            }],
            /*retType=*/"::mlir::LogicalResult",
            /*methodName=*/"inference",
            /*args=*/(ins "InferenceParameter&":$param)
        >,
        InterfaceMethod<
            /*desc=*/[{
            }],
            /*retType=*/"::mlir::LogicalResult",
            /*methodName=*/"backward_weight",
            /*args=*/(ins "InferenceParameter&":$param, "InferenceParameter&":$b_param),
            /*methodBody=*/"",
            /*defaultImplementation=*/[{
            return mlir::failure();
        }]

        >,
        InterfaceMethod<
            /*desc=*/[{
            }],
            /*retType=*/"::mlir::LogicalResult",
            /*methodName=*/"init",
            /*args=*/(ins "InferenceParameter&":$param)
        >,
        InterfaceMethod<
            /*desc=*/[{
            }],
            /*retType=*/"void",
            /*methodName=*/"deinit",
            /*args=*/(ins "InferenceParameter&":$param)
        >,
    ];
}
#endif // TPU_MLIR_INFERENCEINTERFACE