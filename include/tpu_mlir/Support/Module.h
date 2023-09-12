//
// Created by 王奥博 on 2023/8/28.
//

#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/QuantTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/AttrStruct.h"
#include "tpu_mlir/Support/TensorFile.h"
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
        // init module by ModuleOp in init pass
        void init(ModuleOp module);

        Chip getChip();
        void setChip(Chip chip);
        bool isChip(Chip chip);
        Mode getMode();
        void setMode(Mode mode);
        State getState();

        Platform getPlatform();
        bool isPlatform(Platform plt);

        int64_t getFLOPs();
        void setFLOPs(int64_t flops);
        bool isAsymmetric();
        void setAsymmetric(bool is_asymmetric);

        //-----------------------------------------------------------------
        // Helper Functions for ModuleOp
        //-----------------------------------------------------------------

        ModuleOp getModuleOp();
        Location getLoc();
        MLIRContext *getCtx();

        Value getOriValue(Value v);
        void updateModuleTypes();
        void getNCHW(Value v, int64_t &n, int64_t &c, int64_t &h, int64_t &w,
                     bool left_align = true);
        void getNCHW(llvm::ArrayRef<int64_t> shape, int64_t &n, int64_t &c, int64_t &h,
                     int64_t &w, bool left_align = true);
        void getNCHW(llvm::ArrayRef<int64_t> shape, int64_t &n, int64_t &c, int64_t &h,
                     int64_t &w, group_type_t group_type);
        void getNCHW(Value v, int64_t &n, int64_t &c, int64_t &h, int64_t &w,
                     group_type_t group_type);
        void getNCDHW(Value v, int64_t &n, int64_t &c, int64_t &d, int64_t &h,
                      int64_t &w, group_type_t group_type);
        int64_t getNumElements(Value v);
        Type getStorageType(Value v); // storage type
        Type getStorageType(Type type);
        Type getElementType(Value v);
        void setShape(Value v, llvm::ArrayRef<int64_t> shape);
        llvm::ArrayRef<int64_t> getShape(Value v);
        bool isUnranked(Value v);
        void setShapeOrVerify(Value v, llvm::ArrayRef<int64_t> shape);
        bool isSign(Value v);
        bool isWeight(Value v);
        bool isAllWeight(Operation *op);
        bool isNone(Value v);
        FuncOp getMainFuncOp(ModuleOp module);
        i32_array_t getI32Array(ArrayAttr arrayAttr);
        i32_array_t getI32Array(llvm::Optional<ArrayAttr> arrayAttr, int64_t num_elem,
                                int32_t default_value);
        i64_array_t getI64Array(ArrayAttr arrayAttr);
        i64_array_t getI64Array(llvm::Optional<ArrayAttr> arrayAttr, int64_t num_elem,
                                int64_t default_value);
        f64_array_t getF64Array(ArrayAttr arrayAttr);
        f64_array_t getF64Array(llvm::Optional<ArrayAttr> arrayAttr, int64_t num_elem,
                                double default_value);
        bool isOpInGroup(Operation *Op, int64_t *group_type = nullptr);
        bool isOpInParallel(Operation *Op);
        FuncOp getFuncOp(ModuleOp module, StringRef func_name);
        func::CallOp getCallOp(FuncOp func);
        llvm::StringRef getName(Operation *op, int index = 0);
        llvm::StringRef getName(Value v);
        uint32_t getIdx(Value v);
        NameLoc getLoc(Value v);
        void setLoc(Value v, NameLoc loc);
        NameLoc getLocLike(Operation *op, llvm::StringRef suffix);
        NameLoc getLocLike(Value v, llvm::StringRef suffix);

        bool isTpuOp(Operation *op);
        bool isInt4Op(Operation *op);
        bool isCV18xx();
        bool isBM1684Family();
        bool isBM1684XFamily();
        bool isBM1686();
        bool isBM1684X();


        //-----------------------------------------------------------------
        // Helper Functions for submodule
        //-----------------------------------------------------------------
        int getNumSubModule();
        std::shared_ptr<std::vector<ModuleOp>> getAllModules();

        //-----------------------------------------------------------------
        // Helper Functions for weight
        //-----------------------------------------------------------------
        mlir::TensorFile &weightFile();
        void setWeightFileName(const std::string &name);
        void saveWeight();
        void detachWeightFile();

        //-----------------------------------------------------------------
        // Helper Functions for quantization
        //-----------------------------------------------------------------
        bool isCalibratedType(Type type);
        bool isCalibratedType(Value v);
        template <typename... Args> bool isCalibratedType(Value v, Args... args) {
            return isCalibratedType(v) && isCalibratedType(args...);
        }
        bool isUniformQuantized(Type type);
        bool isUniformQuantized(Value v);
        template <typename... Args> bool isUniformQuantized(Value v, Args... args) {
            return isUniformQuantized(v) && isUniformQuantized(args...);
        }
        quant::UniformQuantizedType getUniformQuantizedType(Value v);
        quant::UniformQuantizedType getUniformQuantizedType(Type t);

        //-----------------------------------------------------------------
        // Helper for shape op inference
        //-----------------------------------------------------------------
        class ShapeHelper {
        private:
            ShapeHelper(){};
            ~ShapeHelper(){};
//            ShapeHelper(const ShapeHelper &);
//            ShapeHelper &operator=(const ShapeHelper &);

        public:
            static ShapeHelper &getInstance() {
                static ShapeHelper instance;
                return instance;
            }

            void bindShapeInfo(const Value &v, const std::vector<int64_t> &shape);
            std::vector<int64_t> getShapeInfo(const Value &v);
            bool isShape(const Value &v);

        private:
            llvm::DenseMap<Value, std::vector<int64_t>> _shape_info;
        };

        void bindShapeTensorValue(const Value &v, const std::vector<int64_t> &shape);
        std::vector<int64_t> getShapeTensorValue(const Value &v);
        bool isShape(const Value &v);

    } // namespace module
} // namespace tpu_mlir