file(GLOB _sources
        *.cpp)

add_llvm_library(TPUMLIRBackend
        ${_sources}
        LINK_LIBS PUBLIC
        TPUMLIRSupport
        )