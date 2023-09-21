//
// Created by 王奥博 on 2023/8/28.
//
#include "tpu_mlir/Backend/Arch.h"
#include "mlir/IR/Builders.h"
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "llvm/Support/DynamicLibrary.h"
#include "tpu_mlir/Support/TensorFile.h"

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

        static ModuleOp m = nullptr;
        static MLIRContext *ctx = nullptr;
        static Chip chip = Chip::ALL;
        static Platform platform = Platform::ONNX;
        static std::string weightFileName = "";
        static std::unique_ptr<mlir::TensorFile> wFile = nullptr;

        void init(ModuleOp module) {
            m = module;
            ctx = m->getContext();
            auto chip_ = m->getAttrOfType<StringAttr>(Attr::CHIP);
            chip = symbolizeChip(chip_).value_or(Chip::ALL);
            wFile = nullptr;
            if (m->hasAttrOfType<StringAttr>(Attr::PLATFORM)) {
                auto p = m->getAttrOfType<StringAttr>(Attr::PLATFORM);
                platform = symbolizePlatform(p).value_or(Platform::ONNX);
            } else {
                platform = Platform::ONNX;
            }
        }

        Chip getChip() {
            return chip;
        }

        void setChip(Chip chip_) {
            chip = chip_;
            auto s = stringifyChip(chip_);
            m->setAttr(Attr::CHIP, StringAttr::get(m.getContext(), s));
        }

        bool isChip(Chip chip_) {
            return chip == chip_;
        }

        Mode getMode() {
            return Mode::BF16;
        }

        void setMode(Mode mode) {
            auto s = stringifyMode(mode);
            m->setAttr(Attr::MODE, StringAttr::get(ctx, s));
        }

        State getState() {
            auto s = m->getAttrOfType<StringAttr>(Attr::STATE);
            return symbolizeState(s).value_or(State::TOP_F32);
        }

        void setState(State state) {
            auto s = stringifyState(state);
            m->setAttr(Attr::STATE, StringAttr::get(ctx, s));
        }

        bool isState(State state) {
            return false;
        }

        Platform getPlatform() {
            return platform;
        }

        bool isPlatform(Platform plt) {
            return platform == plt;
        }

        int64_t getFLOPs() {
            return 0;
        }

        void setFLOPs(int64_t flops) {

        }

        bool isAsymmetric() {
            if (m->hasAttrOfType<BoolAttr>(Attr::ASYMMETRIC)) {
                return m->getAttrOfType<BoolAttr>(Attr::ASYMMETRIC).getValue();
            }
            return false;
        }

        void setAsymmetric(bool is_asymmetric) {
            m->setAttr(Attr::ASYMMETRIC, BoolAttr::get(ctx, is_asymmetric));
        }

        ModuleOp getModuleOp() {
            return m;
        }

        Location getLoc() {
            return m.getLoc();
        }

        MLIRContext *getCtx() {
            return ctx;
        }

        static ModuleOp getModuleOp(Value v) {
            auto parent_op = v.getParentBlock()->getParentOp();
            while (parent_op != nullptr && !isa<ModuleOp>(parent_op)) {
                parent_op = parent_op->getParentOp();
            }
            if (parent_op == nullptr) {
                return nullptr;
            }
            return cast<ModuleOp>(parent_op);
        }

        static ModuleOp getModuleOp(Operation *op) {
            while (op != nullptr && !isa<ModuleOp>(op)) {
                op = op->getParentOp();
            }
            if (op == nullptr) {
                return nullptr;
            }
            return cast<ModuleOp>(op);
        }

        Value getOriValue(Value v) {
            auto s = getModuleOp(v);
            if (!s) {
                return v;
            }
            if (auto block_arg = v.dyn_cast_or_null<BlockArgument>()) {
                int idx = block_arg.getArgNumber();
                // blockargument have multi-layers nest.
                FuncOp func_op;
                if (isa<FuncOp>(v.getParentBlock()->getParentOp()))
                    func_op = cast<FuncOp>(v.getParentBlock()->getParentOp());
                else if (isa<tpu::LoopOp, tpu::IfOp, top::LoopOp, top::IfOp>
                        (v.getParentBlock()->getParentOp())) {
                    return getOriValue(v.getParentBlock()->getParentOp()->getOperand(idx));
                } else
                    func_op = v.getParentBlock()->getParentOp()->getParentOfType<FuncOp>();

                if (func_op) {
                    // cur call op
                    auto call_op = getCallOp(func_op);
                    // pre call op
                    auto operand = call_op.getOperand(idx);
                    if (operand.isa<BlockArgument>()) {
                        auto find_root = [](auto &&Me, Value v) -> Value {
                            if (v.isa<BlockArgument>()) {
                                int index = dyn_cast<BlockArgument>(v).getArgNumber();
                                FuncOp func_op;
                                if (isa<FuncOp>(v.getParentBlock()->getParentOp()))
                                    func_op = cast<FuncOp>(v.getParentBlock()->getParentOp());
                                else
                                    func_op =
                                            v.getParentBlock()->getParentOp()->getParentOfType<FuncOp>();
                                auto call_op = getCallOp(func_op);
                                return Me(Me, call_op.getOperand(index));
                            } else {
                                return v;
                            }
                        };

                        Value src_v = find_root(find_root, operand);
                        return src_v;
                    }
                    auto result = operand.cast<OpResult>();
                    auto opd = result.getDefiningOp();
                    if (isa<top::InputOp>(opd)) {
                        return operand;
                    }
                    auto pre_call_op = dyn_cast<func::CallOp>(opd);
                    auto pre_func_op = getFuncOp(s, pre_call_op.getCallee());
                    auto return_op = dyn_cast<ReturnOp>(pre_func_op.front().back());
                    return return_op.getOperand(result.getResultNumber());
                }
            } else if (auto pre_op = v.getDefiningOp()) {
                if (isa<func::CallOp>(pre_op)) {
                    auto call_op = dyn_cast<func::CallOp>(pre_op);
                    int index = v.cast<OpResult>().getResultNumber();
                    for (auto func : s.getOps<FuncOp>()) {
                        if (call_op.getCallee() == func.getName()) {
                            Block &entryBlock = func.front();
                            auto returnOp = dyn_cast<ReturnOp>(entryBlock.back()).getOperation();
                            return returnOp->getOperand(index);
                        }
                    }
                } else {
                    return v;
                }
            }

            llvm_unreachable("Failed to get preOperation.FIx me");
        }

        FuncOp getMainFuncOp(ModuleOp module) { return getFuncOp(module, "main"); }

        static void updateModuleTypes(ModuleOp s) {
            Builder builder(ctx);
            // update callee func's return types
            for (auto func : s.getOps<FuncOp>()) {
                if (func.getName() == "main") {
                    continue;
                }
                std::vector<Type> returns;
                Block &entryBlock = func.front();
                auto returnOp = dyn_cast<ReturnOp>(entryBlock.back()).getOperation();
                for (uint32_t i = 0; i < returnOp->getNumOperands(); ++i) {
                    returns.push_back(returnOp->getOperand(i).getType());
                }
                auto fnType = builder.getFunctionType(func.getArgumentTypes(),
                                                      llvm::ArrayRef<Type>{returns});
                func.setType(fnType);
                auto callee = getCallOp(func);
                if (callee) {
                    for (auto it : llvm::zip(callee.getResults(), returns)) {
                        std::get<0>(it).setType(std::get<1>(it));
                    }
                }
            }
            // update callee arg types
            for (auto func : s.getOps<FuncOp>()) {
                if (func.getName() == "main") {
                    continue;
                }
                auto callee = getCallOp(func);
                if (!callee) {
                    continue;
                }
                std::vector<Type> arguments;
                for (auto it :
                        llvm::zip(callee.getOperandTypes(), func.front().getArguments())) {
                    arguments.push_back(std::get<0>(it));
                    std::get<1>(it).setType(std::get<0>(it));
                }
                auto fnType = builder.getFunctionType(llvm::ArrayRef<Type>(arguments),
                                                      func.getResultTypes());
                func.setType(fnType);
            }
            // update main op return types
            auto mainFunc = getMainFuncOp(s);
            Block &entryBlock = mainFunc.front();
            auto returnOp = dyn_cast<ReturnOp>(entryBlock.back()).getOperation();
            std::vector<Type> returns;
            for (uint32_t i = 0; i < returnOp->getNumOperands(); ++i) {
                returns.push_back(returnOp->getOperand(i).getType());
            }
            auto fnType = builder.getFunctionType(mainFunc.getArgumentTypes(),
                                                  llvm::ArrayRef<Type>{returns});
            mainFunc.setType(fnType);
        }

        void updateModuleTypes() {
            auto modules = getAllModules();
            for (auto s : *modules) {
                updateModuleTypes(s);
            }
        }

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

        void getNCDHW(Value v, int64_t &n, int64_t &c, int64_t &d, int64_t &h, int64_t &w, group_type_t group_type) {
            auto shape = v.getType().cast<RankedTensorType>().getShape();
            int num_dims = shape.size();
            if (GROUP_3D == group_type) {
                n = num_dims > 0 ? shape[0] : 1;
                c = num_dims > 1 ? shape[1] : 1;
                d = num_dims > 2 ? shape[2] : 1;
                h = num_dims > 3 ? shape[3] : 1;
                w = 1;
                for (size_t i = 4; i < num_dims; ++i) {
                    w *= shape[i];
                }
                return;
            } else if (GROUP_MM_INT4 == group_type) {
                assert(num_dims == 2);
                n = shape[0];
                c = 1;
                d = 1;
                h = shape[1];
                w = 1;
            } else {
                d = 1;
                getNCHW(shape, n, c, h, w, group_type);
            }
        }

        int64_t getNumElements(Value v) {
            if (v.getType().isa<RankedTensorType>() == false) {
                return 0;
            }
            auto type = v.getType().cast<RankedTensorType>();
            return type.getNumElements();
        }

        Type getStorageType(Value v) {
            return getStorageType(v.getType());
        }

        Type module::getStorageType(Type type) {
            if (type.isa<RankedTensorType>()) {
                type = type.cast<RankedTensorType>().getElementType();
            }
            if (auto qType = type.dyn_cast<quant::CalibratedQuantizedType>()) {
                return qType.getExpressedType();
            } else if (auto qType = type.dyn_cast<quant::UniformQuantizedType>()) {
                auto stype = qType.getStorageType();
                bool isSign = qType.isSigned();
                if (stype.isSignlessInteger()) {
                    auto bits = stype.getIntOrFloatBitWidth();
                    auto sign = isSign ? IntegerType::Signed : IntegerType::Unsigned;
                    return IntegerType::get(type.getContext(), bits, sign);
                }
                return stype;
            } else if (auto qType = type.dyn_cast<quant::UniformQuantizedPerAxisType>()) {
                return qType.getStorageType();
            }
            return type;
        }



        void setShape(Value v, llvm::ArrayRef<int64_t> shape) {
            auto newType = RankedTensorType::get(shape, getElementType(v));
            v.setType(newType);
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

        bool isSign(Value v) {
            auto stype = getStorageType(v);
            if (stype.isUnsignedInteger()) {
                return false;
            }
            return true;
        }

        bool isWeight(Value v) {
            auto op = v.getDefiningOp();
            if (op == nullptr) {
                return false;
            }
            if (isa<top::WeightOp>(op)) {
                return true;
            }
            return false;
        }

        bool isAllWeight(Operation *op) {
            for (auto in : op->getOperands()) {
                if (isNone(in) || isWeight(in)) {
                    continue;
                }
                return false;
            }
            return true;
        }

        bool isNone(Value v) {
            return v.getType().isa<mlir::NoneType>();
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

        bool isOpInGroup(Operation *Op, int64_t *group_type) {
            if (Op == nullptr) {
                return false;
            }
            auto parent = Op->getParentOp();
            if (isa_and_nonnull<tpu::GroupOp>(parent)) {
                if (group_type) {
                    if (auto groupop = dyn_cast<tpu::GroupOp>(Op)) {
                        *group_type = groupop.getGroupType();
                    }
                }
                return true;
            }
            return false;
        }

        bool isOpInParallel(Operation *Op) {
            if (Op == nullptr) {
                return false;
            }
            auto parent = Op->getParentOp();
            if (isa_and_nonnull<tpu::ParallelOp>(parent)) {
                return true;
            }
            return false;
        }

        FuncOp getFuncOp(ModuleOp mod, StringRef func_name) {
            for (auto func : mod.getOps<FuncOp>()) {
                if (func.getName() == func_name) {
                    return func;
                }
            }
            llvm::errs() << "Can't find FuncOp:" << func_name << "\n";
            llvm_unreachable("Error getFuncOp !!\n");
            return nullptr;
        }

        func::CallOp getCallOp(FuncOp func) {
            auto parent = func->getParentOp();
            auto s = cast<ModuleOp>(parent);
            func::CallOp call = nullptr;
            for (auto each_func : s.getOps<FuncOp>()) {
                WalkResult result =
                        each_func.walk<WalkOrder::PreOrder>([&](func::CallOp op) {
                            if (!call && op.getCallee() == func.getName()) {
                                call = op;
                                return WalkResult::interrupt();
                            }
                            return WalkResult::advance();
                        });
                if (result.wasInterrupted())
                    break;
            }
            return call;
        }

        llvm::StringRef getName(Operation *op, int index) {
            if (auto module = dyn_cast<ModuleOp>(op)) {
                return module.getName().value_or("Unknown");
            }
            if (auto loc = op->getLoc().dyn_cast<NameLoc>()) {
                return loc.getName();
            }
            if (auto loc = op->getLoc().dyn_cast<FusedLoc>()) {
                auto locs = loc.getLocations();
                assert(index < locs.size());
                if (auto name_loc = locs[index].dyn_cast<NameLoc>()) {
                    return name_loc.getName();
                }
            }
            op->print(llvm::errs(), OpPrintingFlags().useLocalScope().enableDebugInfo());
            llvm::errs() << "\n";
            llvm_unreachable("op has no name location!!!");
            return "";
        }

        llvm::StringRef getName(Value v) {
            return getLoc(v).getName().strref();
        }

        uint32_t getIdx(Value v) {
            uint32_t idx = 0;
            if (auto r = v.dyn_cast<OpResult>()) {
                idx = r.getResultNumber();
            } else if (auto r = v.dyn_cast<BlockArgument>()) {
                idx = r.getArgNumber();
            } else {
                v.dump();
                llvm_unreachable("Not Implemented");
            }
            return idx;
        }

        NameLoc getLoc(Value v) {
            if (auto loc = v.getLoc().dyn_cast<NameLoc>()) {
                return loc;
            } else if (auto fuse_loc = v.getLoc().dyn_cast<FusedLoc>()) {
                auto locs = fuse_loc.getLocations();
                uint32_t idx = getIdx(v);
                if (auto name_loc = locs[idx].dyn_cast<NameLoc>()) {
                    return name_loc;
                }
            } else if (auto op = v.getDefiningOp()) {
                auto loc = op->getLoc();
                if (auto name_loc = loc.dyn_cast<NameLoc>()) {
                    return name_loc;
                }
                if (auto fuse_loc = loc.dyn_cast<FusedLoc>()) {
                    uint32_t idx = getIdx(v);
                    auto locs = fuse_loc.getLocations();
                    if (auto name_loc = locs[idx].dyn_cast<NameLoc>()) {
                        return name_loc;
                    }
                }
            }
            v.dump();
            llvm_unreachable("Not Implemented");
            return nullptr;
        }

        void setLoc(Value v, NameLoc loc) {
            if (v.getLoc().isa<NameLoc>()) {
                v.setLoc(loc);
                return;
            }
            if (auto fuse_loc = v.getLoc().dyn_cast<FusedLoc>()) {
                std::vector<mlir::Location> locs = fuse_loc.getLocations();
                uint32_t idx = getIdx(v);
                locs[idx] = loc;
                auto new_loc = FusedLoc::get(v.getContext(), locs);
                v.setLoc(new_loc);
                return;
            }
            if (auto op = v.getDefiningOp()) {
                auto op_loc = op->getLoc();
                if (op_loc.isa<NameLoc>()) {
                    op->setLoc(loc);
                    return;
                }
                if (auto fuse_loc = op->getLoc().dyn_cast<FusedLoc>()) {
                    std::vector<mlir::Location> locs = fuse_loc.getLocations();
                    auto idx = getIdx(v);
                    locs[idx] = loc;
                    auto new_loc = FusedLoc::get(v.getContext(), locs);
                    op->setLoc(new_loc);
                    return;
                }
            }
            v.dump();
            llvm_unreachable("Not Implemented");
        }

        NameLoc getLocLike(Operation *op, llvm::StringRef suffix) {
            return getLocLike(op->getResult(0), suffix);
        }

        NameLoc module::getLocLike(Value v, llvm::StringRef suffix) {
            auto name = getName(v);
            auto new_name = name.str() + "_" + suffix.str();
            Builder builder(v.getContext());
            return NameLoc::get(builder.getStringAttr(new_name));
        }

        bool isTpuOp(Operation *op) {
            return (op->getDialect()->getNamespace() == "tpu");
        }

        bool isInt4Op(Operation *op) {
            return false;
        }

        bool isCV18xx() {
            return (chip == Chip::CV183x || chip == Chip::CV182x ||
                    chip == Chip::CV181x || chip == Chip::CV180x);
        }

        bool isBM1684Family() {
            return (chip == Chip::BM1684);
        }

        bool isBM1684XFamily() {
            return (chip == Chip::BM1684X || chip == Chip::BM1686 || chip == Chip::CV186X);
        }

        bool isBM1686() {
            return (chip == Chip::BM1686 || chip == Chip::CV186X);
        }

        bool isBM1684X() {
            return chip == Chip::BM1684X;
        }

        int getNumSubModule() {
            auto sub = m.getOps<ModuleOp>();
            return std::distance(sub.begin(), sub.end());
        }

        std::shared_ptr<std::vector<ModuleOp>> getAllModules() {
            auto modules = std::make_shared<std::vector<ModuleOp>>();
            auto sub = m.getOps<ModuleOp>();
            if (sub.empty()) {
                modules->push_back(m);
            } else {
                modules->assign(sub.begin(), sub.end());
            }
            return std::move(modules);
        }

        mlir::TensorFile &weightFile() {
            if (wFile == nullptr) {
                auto name = m->getAttrOfType<StringAttr>(Attr::WEIGHT_FILE).getValue();
                wFile = std::make_unique<mlir::TensorFile>(name, false);
            }
            return *wFile;
        }

        void setWeightFileName(const std::string &name) {
            weightFileName = name;
        }

        //-----------------------------------------------------------------
        // Helper Functions for weight
        //-----------------------------------------------------------------
        static std::string genWeightFileName(bool &same_name) {
            auto name = getName(m);
            auto state = getState();
            auto chip_ = getChip();
            auto chip = stringifyChip(chip_);
            auto old_name = m->getAttrOfType<StringAttr>(Attr::WEIGHT_FILE).getValue();
            std::string file_name = name.lower() + std::string("_") +
                                    stringifyState(state).lower() + std::string("_") +
                                    chip.lower();
            if (!isChip(Chip::ALL)) {
                auto mode = getMode();
                std::string sym = "";
                if (mode == Mode::INT8) {
                    sym = isAsymmetric() ? "_asym" : "_sym";
                }
                auto mode_ = stringifyMode(mode);
                file_name += std::string("_") + mode_.lower() + sym;
            }
            auto new_name = file_name + "_weight.npz";
            same_name = (old_name == new_name);
            if (same_name) {
                new_name = file_name + "_weight_fix.npz";
            }
            return new_name;
        }

        void saveWeight() {
// check name conflict
            std::set<StringRef> all_names;
            auto modules = getAllModules();
            for (auto s : *modules) {
                for (auto func : s.getOps<FuncOp>()) {
                    func.walk([&](Operation *op) {
                        if (op->getLoc().dyn_cast<NameLoc>() && !module::isOpInGroup(op) &&
                            !module::isOpInParallel(op) &&
                            !isa<func::ReturnOp, func::CallOp, func::FuncOp, tpu::YieldOp,
                                    tpu::IfOp, top::InputOp>(op)) {
                            auto name = module::getName(op);
                            // if op have more than two regions, it can have the same op Name
                            if (all_names.find(name) != all_names.end()) {
                                op->dump();
                                llvm_unreachable("op name conflict");
                            }
                            all_names.insert(name);
                        }
                    });
                }
            }
            bool same_name = true;
            std::string filename_;
            if (weightFileName == "") {
                filename_ = module::genWeightFileName(same_name);
            } else {
                same_name = false;
                filename_ = weightFileName;
            }
            // weight remove unused in npz
            if (wFile == nullptr) {
                if (!same_name) {
                    weightFile().save(filename_);
                    m->setAttr(Attr::WEIGHT_FILE, StringAttr::get(ctx, filename_));
                }
                return;
            }
            if (wFile->changed() == false && same_name) {
                return;
            }
            std::set<StringRef> weight_names;
            for (auto s : *modules) {
                for (auto func : s.getOps<FuncOp>()) {
                    func.walk([&](top::WeightOp op) {
                        weight_names.insert(module::getName(op.getOperation()));
                    });
                }
            }
            std::set<StringRef> npz_names;
            wFile->getAllNames(npz_names);
            std::set<StringRef> dif_names;
            for (auto name : npz_names) {
                if (weight_names.find(name) == weight_names.end()) {
                    dif_names.insert(name);
                }
            }
            for (auto &name : dif_names) {
                wFile->deleteTensor(name);
            }
            if (wFile->changed() == false && same_name) {
                return;
            }
            wFile->save(filename_);
            m->setAttr(Attr::WEIGHT_FILE, StringAttr::get(ctx, filename_));
        }

        void detachWeightFile() {
            wFile = nullptr;
        }

        bool module::isCalibratedType(Type type) {
            return type.cast<RankedTensorType>()
                    .getElementType()
                    .isa<quant::CalibratedQuantizedType>();
        }

        bool module::isCalibratedType(Value v) {
            return isCalibratedType(v.getType());
        }

        bool module::isUniformQuantized(Type type) {
            if (type.isa<RankedTensorType>() == false) {
                return false;
            }
            return type.cast<RankedTensorType>()
                    .getElementType()
                    .isa<quant::UniformQuantizedType>();
        }

        bool module::isUniformQuantized(Value v) {
            return isUniformQuantized(v.getType());
        }

        quant::UniformQuantizedType getUniformQuantizedType(Value v) {
            return v.getType()
                    .cast<RankedTensorType>()
                    .getElementType()
                    .cast<quant::UniformQuantizedType>();
        }

        quant::UniformQuantizedType module::getUniformQuantizedType(Type t) {
            return t.cast<RankedTensorType>()
                    .getElementType()
                    .cast<quant::UniformQuantizedType>();
        }

        void ShapeHelper::bindShapeInfo(const Value &v, const std::vector<int64_t> &shape) {
            _shape_info[v] = shape;
        }

        std::vector<int64_t> ShapeHelper::getShapeInfo(const Value &v) {
            return _shape_info.at(v);
        }

        bool ShapeHelper::isShape(const Value &v) {
            return _shape_info.find(v) != _shape_info.end();
        }

        void bindShapeTensorValue(const Value &v, const std::vector<int64_t> &shape) {
            ShapeHelper::getInstance().bindShapeInfo(v, shape);
        }

        std::vector<int64_t> getShapeTensorValue(const Value &v) {
            return ShapeHelper::getInstance().getShapeInfo(v);
        }

        bool isShape(const Value &v) {
            return ShapeHelper::getInstance().isShape(v);
        }
    } // namespace module
} // namespace tpu_mlir