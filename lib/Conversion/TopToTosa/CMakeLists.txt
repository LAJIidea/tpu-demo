file(GLOB _sources
        *.cpp)

add_mlir_conversion_library(TPUMLIRTopToTosa
        ${_sources}

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/tpu_mlir/Dialect/Top
        ${MLIR_INCLUDE_DIRS}/Dialect/Tosa

        DEPENDS
        TpuMlirConversionPassIncGen

        LINK_LIBS PUBLIC
        TPUMLIRTop
        MLIRTosaDialect)