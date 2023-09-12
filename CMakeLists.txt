cmake_minimum_required(VERSION 3.24)

if (POLICY CMP0116)
    cmake_policy(SET CMP0116 OLD)
endif()

project(tpu_demo LANGUAGES CXX C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

#-------------------------------------------------------------------------------
# MLIR/LLVM Configuration
#-------------------------------------------------------------------------------
#set(LLVM_DIR "/Volumes/T7/workspace/lib/llvm/lib/cmake/llvm")
#set(MLIR_DIR "/Volumes/T7/workspace/lib/llvm/lib/cmake/mlir")
set(LLVM_DIR "E://workspace//open-source//lib//llvm//lib//cmake//llvm")
set(MLIR_DIR "E://workspace//open-source//lib//llvm//lib//cmake//mlir")
find_program(CCACHE ccache)
if(CCACHE)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ${CCACHE})
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ${CCACHE})
endif()

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
#include(HandleLLVMOptions)

#-------------------------------------------------------------------------------
# zlib Configuration
#-------------------------------------------------------------------------------
find_package(ZLIB REQUIRED)
if(ZLIB_FOUND)
    message(STATUS "Found zlib library: ${ZLIB_LIBRARIES}")
else()
    message(FATAL_ERROR "zlib library not found")
endif()
#-------------------------------------------------------------------------------
# oneDNN Configuration
#-------------------------------------------------------------------------------
#set(DNNL_PATH "/Volumes/T7/workspace/lib/oneDNN")
set(DNNL_PATH "E://workspace//open-source//lib//oneDNN")
set(DNNL_DIR ${DNNL_PATH}/lib/cmake/dnnl)
find_package(DNNL REQUIRED CONFIG)
include_directories(${DNNL_PATH}/include)
link_directories(${DNNL_PATH}/lib)
#-------------------------------------------------------------------------------
# cnpy Configuration
#-------------------------------------------------------------------------------
set(CNPY_PATH ${PROJECT_SOURCE_DIR}/third_party/cnpy)
include_directories(${CNPY_PATH})

OPTION (USE_OpenMP "Use OpenMP" ON)
IF(USE_OpenMP)
    FIND_PACKAGE(OpenMP)
    IF(OPENMP_FOUND)
        message("enter here")
        SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    ENDIF()
ENDIF()


include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(${CMAKE_INSTALL_PREFIX}/include)
message(STATUS "Using LLVM_INCLUDE_DIRS in:" ${LLVM_INCLUDE_DIRS})
message(STATUS "Using MLIR_INCLUDE_DIRS in: ${MLIR_INCLUDE_DIRS}")

if (WIN32)
else ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Werror -Wno-unused-result -Wreturn-type -Wunused-variable")
    message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
endif ()
add_subdirectory(include)
add_subdirectory(third_party)
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(unitest)

