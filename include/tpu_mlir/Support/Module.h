//
// Created by 王奥博 on 2023/8/28.
//

#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/QuantTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/AttrStruct.h"
#include "tpu_mlir/Support/ModuleEnum.h.inc"

using namespace mlir;
using namespace mlir::func;
using namespace tpu_mlir;

namespace tpu_mlir {

    typedef enum {
        GROUP_NORMAL = 0,
        GROUP_3D = 1,
        GROUP_SMALL_C = 2,
        GROUP_MM_INT4 = 3,
        GROUP_MM = 4,
        GROUP_UNSUPPORT
    } group_type_t;

    //-----------------------------------------------------------------
    // Types
    //-----------------------------------------------------------------
    typedef std::shared_ptr<std::vector<int32_t>> i32_array_t;
    typedef std::shared_ptr<std::vector<int64_t>> i64_array_t;
    typedef std::shared_ptr<std::vector<double>> f64_array_t;

    namespace module {
        void getNCHW(Value v, int64_t &n, int64_t &c, int64_t &h, int64_t &w,
                     bool left_align = true);
        void getNCHW(llvm::ArrayRef<int64_t> shape, int64_t &n, int64_t &c, int64_t &h,
                     int64_t &w, bool left_align = true);
        void getNCHW(llvm::ArrayRef<int64_t> shape, int64_t &n, int64_t &c, int64_t &h,
                     int64_t &w, group_type_t group_type);
        void getNCHW(Value v, int64_t &n, int64_t &c, int64_t &h, int64_t &w,
                     group_type_t group_type);
        int64_t getNumElements(Value v);
        llvm::ArrayRef<int64_t> getShape(Value v);
        bool isUnranked(Value v);
        void setShapeOrVerify(Value v, llvm::ArrayRef<int64_t> shape);
        i32_array_t getI32Array(ArrayAttr arrayAttr);
        i32_array_t getI32Array(llvm::Optional<ArrayAttr> arrayAttr, int64_t num_elem,
                                int32_t default_value);
        i64_array_t getI64Array(ArrayAttr arrayAttr);
        i64_array_t getI64Array(llvm::Optional<ArrayAttr> arrayAttr, int64_t num_elem,
                                int64_t default_value);
        f64_array_t getF64Array(ArrayAttr arrayAttr);
        f64_array_t getF64Array(llvm::Optional<ArrayAttr> arrayAttr, int64_t num_elem,
                                double default_value);

    } // namespace module
} // namespace tpu_mlir