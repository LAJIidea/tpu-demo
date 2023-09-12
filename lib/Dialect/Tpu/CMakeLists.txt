file(GLOB _sources
        IR/*.cpp
        Interfaces/Common/*.cpp
        Interfaces/BM1684X/*.cpp
        Interfaces/BM1684/*.cpp
        Interfaces/CV18xx/*.cpp)

add_llvm_library(TPUMLIRTpu
        ${_sources}

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/tpu_mlir/Dialect/Top
        ${PROJECT_SOURCE_DIR}/include/tpu_mlir/Dialect/Tpu

        DEPENDS
        MLIRTopOpsIncGen
        MLIRTpuOpsIncGen
        MLIRTpuAttrGen
        MLIRInferenceInterfaceIncGen
        MLIRGlobalGenInterfaceIncGen
        MLIRLocalGenInterfaceIncGen
        MLIRDynLocalGenInterfaceIncGen
        MLIRDynGlobalGenInterfaceIncGen
        MLIRTypeInterfaceIncGen
        MLIRIndexingMapsInterfaceIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRPass
        MLIRSupport
        MLIRParser
        MLIRFuncDialect
        MLIRQuantDialect
        MLIRRewrite
        MLIRTransforms
        MLIRInferenceInterface
        MLIRGlobalGenInterface
        MLIRLocalGenInterface
        MLIRDynLocalGenInterface
        MLIRDynGlobalGenInterface
        MLIRIndexingMapsInterface
        MLIRTypeInterface
        TPUMLIRSupport
        TPUMLIRTrait
        TPUMLIRBackend
        )