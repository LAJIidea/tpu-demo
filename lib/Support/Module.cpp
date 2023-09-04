//
// Created by 王奥博 on 2023/8/28.
//
#include "tpu_mlir/Backend/Arch.h"
#include "mlir/IR/Builders.h"
#include "tpu_mlir/Support/Module.h"
#include "llvm/Support/DynamicLibrary.h"

#include "tpu_mlir/Support/ModuleEnum.cpp.inc"

namespace tpu_mlir {
    namespace module {
        struct Attr {
            static constexpr llvm::StringRef NAME = "model.name";
            static constexpr llvm::StringRef STATE = "module.state";
            static constexpr llvm::StringRef CHIP = "module.chip";
            static constexpr llvm::StringRef WEIGHT_FILE = "module.weight_file";
            static constexpr llvm::StringRef FLOPS = "module.FLOPs";
            static constexpr llvm::StringRef CORES = "module.cores";
            static constexpr llvm::StringRef COEFF_ADDR = "module.coeff_addr";
            static constexpr llvm::StringRef COEFF_SIZE = "module.coeff_size";
            static constexpr llvm::StringRef NEURON_ADDR = "module.neuron_addr";
            static constexpr llvm::StringRef NEURON_SIZE = "module.neuron_size";
            static constexpr llvm::StringRef GMEM_PRIVATE_SIZE = "module.private_size";
            static constexpr llvm::StringRef ASYMMETRIC = "module.asymmetric";
            static constexpr llvm::StringRef MODE = "module.mode";
            static constexpr llvm::StringRef PLATFORM = "module.platform";
            static constexpr llvm::StringRef POSTPROCESS = "module.postprocess";
        };

//        static ModuleOp m = nullptr;
//        static MLIRContext *ctx = nullptr;
//        static Chip chip = Chip::ALL;
//        static Platform platform = Platform::ONNX;
//        static std::string weightFileName = "";

        Type getElementType(Value v) {
            auto type = v.getType();
            if (type.isa<RankedTensorType>()) {
                auto rtype = type.cast<RankedTensorType>();
                return rtype.getElementType();
            } else if (type.isa<UnrankedTensorType>()) {
                auto rtype = type.cast<UnrankedTensorType>();
                return rtype.getElementType();
            }
            return type;
        }

        static void getNCHW_align_right(llvm::ArrayRef<int64_t> &shape, int64_t &n,
                                        int64_t &c, int64_t &h, int64_t &w) {
            int num_dims = shape.size();
            n = 1, c = 1, h = 1, w = 1;
            if (num_dims > 0) {
                w = shape[num_dims - 1];
            }
            if (num_dims > 1) {
                h = shape[num_dims - 2];
            }
            if (num_dims > 2) {
                c = shape[num_dims - 3];
            }
            if (num_dims > 3) {
                n = shape[num_dims - 4];
            }
            for (int i = 4; i < num_dims; i++) {
                n *= shape[num_dims - i - 1];
            }
        }

        static void getNCHW_align_left(llvm::ArrayRef<int64_t> shape, int64_t &n,
                                       int64_t &c, int64_t &h, int64_t &w) {
            int num_dims = shape.size();
            n = 1, c = 1, h = 1, w = 1;
            if (num_dims > 0) {
                n = shape[0];
            }
            if (num_dims > 1) {
                c = shape[1];
            }
            if (num_dims > 2) {
                h = shape[2];
            }
            for (size_t i = 3; i < num_dims; ++i) {
                w *= shape[i];
            }
        }

        void getNCHW(Value v, int64_t &n, int64_t &c, int64_t &h, int64_t &w, bool left_align) {
            auto shape = v.getType().cast<RankedTensorType>().getShape();
            getNCHW(shape, n, c, h, w, left_align);
        }

        void
        getNCHW(llvm::ArrayRef<int64_t> shape, int64_t &n, int64_t &c, int64_t &h, int64_t &w, bool left_align) {
            if (left_align) {
                getNCHW_align_left(shape, n, c, h, w);
            } else {
                getNCHW_align_right(shape, n, c, h, w);
            }
        }

        void getNCHW(llvm::ArrayRef<int64_t> shape, int64_t &n, int64_t &c, int64_t &h, int64_t &w,
                             group_type_t group_type) {
            if (group_type == GROUP_NORMAL) {
                module::getNCHW(shape, n, c, h, w, true);
            } else if (group_type == GROUP_SMALL_C) {
                int64_t npu_num = backend::Arch::NPU_NUM;
                auto shape_vec = shape.vec();
                shape_vec.resize(4);
                // shape.size() == 2/1 is for MatMul weight and bias
                if (shape.size() == 2) {
                    shape_vec[3] = 1;
                    shape_vec[2] = shape[1];
                    shape_vec[1] = shape[0];
                    shape_vec[0] = 1;
                } else if (shape.size() == 1) {
                    shape_vec[3] = 1;
                    shape_vec[2] = shape[0];
                    shape_vec[1] = 1;
                    shape_vec[0] = 1;
                } else if (shape.size() == 4) {
                    shape_vec[3] = 1;
                    shape_vec[2] = shape[3];
                    shape_vec[1] = shape[2];
                    shape_vec[0] = shape[1] * shape[0];
                    if (shape[2] * shape[1] * shape[0] % npu_num == 0) {
                        shape_vec[1] = npu_num;
                        shape_vec[0] = shape[2] * shape[1] * shape[0] / npu_num;
                    }
                } else if (shape.size() == 5) {
                    shape_vec[3] = 1;
                    shape_vec[2] = shape[4];
                    shape_vec[1] = shape[3];
                    shape_vec[0] = shape[2] * shape[1] * shape[0];
                    if (shape[3] * shape[2] * shape[1] * shape[0] % npu_num == 0) {
                        shape_vec[1] = npu_num;
                        shape_vec[0] = shape[3] * shape[2] * shape[1] * shape[0] / npu_num;
                    }
                }
                module::getNCHW(shape_vec, n, c, h, w, false);
            } else if (GROUP_MM_INT4 == group_type) {
                assert(shape.size() == 2);
                n = shape[0];
                c = 1;
                h = shape[1];
                w = 1;
            } else {
                module::getNCHW(shape, n, c, h, w, true);
            }
        }

        void getNCHW(Value v, int64_t &n, int64_t &c, int64_t &h, int64_t &w, group_type_t group_type) {
            auto shape = v.getType().cast<RankedTensorType>().getShape();
            getNCHW(shape, n, c, h, w, group_type);
        }

        int64_t getNumElements(Value v) {
            if (v.getType().isa<RankedTensorType>() == false) {
                return 0;
            }
            auto type = v.getType().cast<RankedTensorType>();
            return type.getNumElements();
        }

        llvm::ArrayRef<int64_t> getShape(Value v) {
            if (v.getType().isa<NoneType>()) {
                v.dump();
                llvm_unreachable("v is none type");
            }
            if (!isUnranked(v)) {
                auto type = v.getType().cast<RankedTensorType>();
                return type.getShape();
            } else {
                return v.getType().cast<UnrankedTensorType>().getShape();
            }
        }

        bool isUnranked(Value v) {
            return v.getType().isa<mlir::UnrankedTensorType>();
        }

        bool isDynamicShape(Value v) {
            int ret = false;
            auto tensorTy = v.getType().dyn_cast<RankedTensorType>();
            if (tensorTy) {
                for (int64_t dim : tensorTy.getShape()) {
                    if (ShapedType::isDynamic(dim) || dim == 0)
                        ret = true;
                }
            }
            return ret;
        }

        void setShapeOrVerify(Value v, llvm::ArrayRef<int64_t> shape) {
            if (isUnranked(v) || isDynamicShape(v)) {
                auto newType = RankedTensorType::get(shape, getElementType(v));
                v.setType(newType);
            } else {
                auto s = getShape(v);
                /* unranked tensor is okay, for example:
                   tensor<*xf32>->tensor<1xf32> */
                if ((std::max(s.size(), shape.size()) > 1) && s != shape) {
                    v.dump();
                    llvm_unreachable("Shape Verify failed");
                }
            }
        }

        i32_array_t getI32Array(ArrayAttr arrayAttr) {
            auto data = std::make_shared<std::vector<int32_t>>();
            for (auto en : llvm::enumerate(arrayAttr)) {
                auto attr = en.value().dyn_cast<IntegerAttr>();
                if (attr) {
                    data->push_back(attr.getInt());
                } else {
                    arrayAttr.dump();
                    llvm_unreachable("not int32_t type");
                }
            }
            return std::move(data);
        }

        i32_array_t getI32Array(llvm::Optional<ArrayAttr> arrayAttr, int64_t num_elem, int32_t default_value) {
            if (arrayAttr.has_value()) {
                auto arr = getI32Array(arrayAttr.value());
                assert(arr->size() == num_elem);
                return std::move(arr);
            }
            return std::make_shared<std::vector<int32_t>>(num_elem, default_value);
        }

        i64_array_t getI64Array(ArrayAttr arrayAttr) {
            auto data = std::make_shared<std::vector<int64_t>>();
            for (auto en : llvm::enumerate(arrayAttr)) {
                auto attr = en.value().dyn_cast<IntegerAttr>();
                if (attr) {
                    data->push_back(attr.getInt());
                } else {
                    arrayAttr.dump();
                    llvm_unreachable("not int64_t type");
                }
            }
            return std::move(data);
        }

        i64_array_t getI64Array(llvm::Optional<ArrayAttr> arrayAttr, int64_t num_elem, int64_t default_value) {
            if (arrayAttr.has_value()) {
                auto arr = getI64Array(arrayAttr.value());
                assert(arr->size() == num_elem);
                return std::move(arr);
            }
            return std::make_shared<std::vector<int64_t>>(num_elem, default_value);
        }

        f64_array_t getF64Array(ArrayAttr arrayAttr) {
            auto data = std::make_shared<std::vector<double>>();
            for (auto en : llvm::enumerate(arrayAttr)) {
                auto attr = en.value().dyn_cast<FloatAttr>();
                data->push_back(attr.getValueAsDouble());
            }
            return std::move(data);
        }

        f64_array_t getF64Array(llvm::Optional<ArrayAttr> arrayAttr, int64_t num_elem, double default_value) {
            if (arrayAttr.has_value()) {
                auto arr = getF64Array(arrayAttr.value());
                assert(arr->size() == num_elem);
                return std::move(arr);
            }
            return std::make_shared<std::vector<double>>(num_elem, default_value);
        }
    } // namespace module
} // namespace tpu_mlir