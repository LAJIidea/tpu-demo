#ifndef TPU_MLIR_TRAITS
#define TPU_MLIR_TRAITS

include "mlir/IR/OpBase.td"

class TPUMLIROpTrait<string name> : NativeOpTrait<name> {
    let cppNamespace = "::tpu_mlir::trait";
}

def ShapeProducer : TPUMLIROpTrait<"ShapeProducer">;
def ShapeConsumer : TPUMLIROpTrait<"ShapeConsumer">;
def SupportPermuteMove : TPUMLIROpTrait<"SupportPermuteMove">;
def SupportFuseRelu : TPUMLIROpTrait<"SupportFuseRelu">;
def SupportEarlyStride : TPUMLIROpTrait<"SupportEarlyStride">;
def SupportElementwise : TPUMLIROpTrait<"SupportElementwise">;
def TpuTypeRestrict: TPUMLIROpTrait<"TpuTypeRestrict">;
def InOutSameShape: TPUMLIROpTrait<"InOutSameShape">;

#endif // TPU_MLIR_TRAITS