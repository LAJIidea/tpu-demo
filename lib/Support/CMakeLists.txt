file(GLOB _sources *.cpp Dnnl/*.cpp)

add_llvm_library(TPUMLIRSupport
        ${_sources}

        DEPENDS
        ModuleAttrGen
        MLIRTopOpsIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRSupport
        MLIRQuantDialect
        MLIRQuantUtils
        MLIRInferenceInterface
        dnnl
        z
)