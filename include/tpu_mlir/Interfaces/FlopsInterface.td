#ifndef TPU_MLIR_FLOPSINTERFACE
#define TPU_MLIR_FLOPSINTERFACE

include "mlir/IR/OpBase.td"

def FlopsInterface : OpInterface<"FlopsInterface"> {
  let description = [{
  }];
  let cppNamespace = "::tpu_mlir";
  let methods = [
      InterfaceMethod<
        /*desc=*/[{
        }],
        /*retType=*/"int64_t",
        /*methodName=*/"getFLOPs",
        /*args=*/(ins)
      >,
  ];
}
#endif // TPU_MLIR_FLOPSINTERFACE