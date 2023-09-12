file(GLOB _sources *.cpp)

add_llvm_library(cnpy
  ${_sources}
)

target_link_libraries(cnpy ${ZLIB_LIBRARIES})