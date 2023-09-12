file(GLOB _sources *.cpp Dnnl/*.cpp)

if (WIN32)
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
            cnpy
            dnnl
            )
else ()
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
            cnpy
            dnnl
            z
            )
endif ()