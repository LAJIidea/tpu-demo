#ifndef TPU_MLIR_TPU_OPS
#define TPU_MLIR_TPU_OPS

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "tpu_mlir/Interfaces/LocalGenInterface.td"
include "tpu_mlir/Interfaces/GlobalGenInterface.td"
include "tpu_mlir/Interfaces/InferenceInterface.td"
include "tpu_mlir/Interfaces/TypeInterface.td"
include "tpu_mlir/Interfaces/DynLocalGenInterface.td"
include "tpu_mlir/Interfaces/DynGlobalGenInterface.td"
include "tpu_mlir/Interfaces/IndexingMapsInterface.td"
include "tpu_mlir/Traits/Traits.td"

// =============================================================================
//
// Defines Tpu Dialect.
//
//===----------------------------------------------------------------------===//

def Tpu_Dialect : Dialect {
    let name = "tpu";
    let summary = "A tpu dialect for the SOPHGO AI chips";
    let cppNamespace = "::tpu_mlir::tpu";
//    let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Tpu Attributes.
//===----------------------------------------------------------------------===//

class Tpu_Attr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Tpu_Dialect, attrName, traits> {
    let mnemonic = attrMnemonic;
}

// A string attribute whose value are one of the values in `cases`.
class AnyStrAttrOf<list<string> cases> : StringBasedAttr<
        CPred<!foldl(
                "$_self.cast<StringAttr>().getValue() == \"" # !head(cases) # "\"",
                !foreach(case, !tail(cases),
                    "$_self.cast<StringAttr>().getValue() == \"" # case # "\""),
                prev, cur, prev # " || " # cur)>,
        "string attribute whose value is " #
        !foldl(/*init*/!head(cases), /*list*/!tail(cases),
            prev, cur, prev # ", or " # cur)>;


def ArgModeAttr: AnyStrAttrOf<["ArgMin","ArgMax"]>;
def CompareModeAttr: AnyStrAttrOf<["Equal","Greater","GreaterOrEqual","Less","LessOrEqual", "NotEqual", "Not", "And"]>;
def ReduceModeAttr: AnyStrAttrOf<["ReduceMin","ReduceMax","ReduceMean","ReduceL2","ReduceL1","ReduceSum","ReduceProd"]>;
def RoiAlignModeAttr: AnyStrAttrOf<["Avg","Max"]>;
def NonZeroOrderAttr: AnyStrAttrOf<["ColMajor","RowMajor"]>;
def DetectionOutputCodeTypeAttr: AnyStrAttrOf<["CORNER", "CENTER_SIZE", "CORNER_SIZE"]>;
def YoloVersionAttr: AnyStrAttrOf<["yolov3", "yolov3_tiny", "yolov3_spp", "yolov4", "yolov5","yolov8"]>;
def MatchTemplateModeAttr: AnyStrAttrOf<["TM_CCOEFF_NORMED", "TM_SQDIFF"]>;

def Tpu_LayerGroupAttr : Tpu_Attr<"LayerGroup", "lg"> {
    let summary = "Structure of layer group parameters";
    let parameters = (ins
    "int64_t":$out_addr,
    "int64_t":$out_size,
    "int64_t":$buffer_addr,
    "int64_t":$buffer_size,
    "bool":$eu_align,
    "DenseI64ArrayAttr":$n_idx,
    "DenseI64ArrayAttr":$n_slice,
    "DenseI64ArrayAttr":$c_idx,
    "DenseI64ArrayAttr":$c_slice,
    "DenseI64ArrayAttr":$d_idx,
    "DenseI64ArrayAttr":$d_slice,
    "DenseI64ArrayAttr":$h_idx,
    "DenseI64ArrayAttr":$h_slice,
    "DenseI64ArrayAttr":$w_idx,
    "DenseI64ArrayAttr":$w_slice,
    "int64_t":$id,
    "int64_t":$stage,
    "int64_t":$group_type
    );
    let assemblyFormat = "`<` struct(params) `>`";
}

def Tpu_DequantMode: I32EnumAttr<"DequantMode",
        "dequant mode supported by DequantOp",
        [
            I32EnumAttrCase<"Normal", 0>,
            I32EnumAttrCase<"TFLite", 1>
        ]>{
    let genSpecializedAttr = 0;
    let cppNamespace = "::tpu_mlir::tpu";
}
def Tpu_DequantModeAttr : EnumAttr<Tpu_Dialect, Tpu_DequantMode, "dq_mode">;

def Tpu_RequantMode: I32EnumAttr<"RequantMode",
        "requant mode supported by RequantOp",
        [
            I32EnumAttrCase<"TFLite_LShift", 0>,
            I32EnumAttrCase<"TFLite", 1>,  // * Multi >> 31 >> shift, == QDM
            I32EnumAttrCase<"MultiplierShift", 2>, // * Multi >> shift
            I32EnumAttrCase<"OnlyShift", 3>, // >> shift
            I32EnumAttrCase<"QDM", 4>       // similar to TFLite
        ]>{
    let genSpecializedAttr = 0;
    let cppNamespace = "::tpu_mlir::tpu";
}
def Tpu_RequantModeAttr : EnumAttr<Tpu_Dialect, Tpu_RequantMode, "rq_mode">;

def Tpu_PaddingMode: I32EnumAttr<"PaddingMode",
        "requant mode supported by PadOp",
        [
            I32EnumAttrCase<"constant", 0>,
            I32EnumAttrCase<"reflect", 1>,
            I32EnumAttrCase<"symmetric", 2>,
            I32EnumAttrCase<"edge", 3>,
        ]>{
    let genSpecializedAttr = 0;
    let cppNamespace = "::tpu_mlir::tpu";
}
def Tpu_PaddingModeAttr : EnumAttr<Tpu_Dialect, Tpu_PaddingMode, "Pad_Mode">;


def Tpu_RoundMode: I32EnumAttr<"RoundMode",
        "round mode supported by Round",
        [
            I32EnumAttrCase<"HalfAwayFromZero", 0>,
            I32EnumAttrCase<"HalfUp", 1>,
            I32EnumAttrCase<"HalfDown", 2>,
            I32EnumAttrCase<"HalfToEven", 3>,
            I32EnumAttrCase<"HalfToOdd", 4>,
            I32EnumAttrCase<"HalfTowardsZero", 5>,
            I32EnumAttrCase<"TowardsZero", 6>,
            I32EnumAttrCase<"Up", 7>,
            I32EnumAttrCase<"Down", 8>
        ]>{
    let genSpecializedAttr = 0;
    let cppNamespace = "::tpu_mlir::tpu";
}
def Tpu_RoundModeAttr : EnumAttr<Tpu_Dialect, Tpu_RoundMode, "round_mode">;

def Tpu_PoolMode: I32EnumAttr<"PoolMode",
        "pooling mode supported by PoolOp",
        [
            I32EnumAttrCase<"Avg", 0>,
            I32EnumAttrCase<"Max", 1>,
        ]>{
    let genSpecializedAttr = 0;
    let cppNamespace = "::tpu_mlir::tpu";
}
def Tpu_PoolModeAttr : EnumAttr<Tpu_Dialect, Tpu_PoolMode, "pool_mode">;

def Tpu_DistributionPattern: I32EnumAttr<"DistributionPattern",
        "Patterns of distribution",
        [
            I32EnumAttrCase<"MatMulColumn", 0>,
            I32EnumAttrCase<"MatMulRow", 1>,
            I32EnumAttrCase<"MatMulMerge", 2>,
            I32EnumAttrCase<"MatMulSliceMerge", 3>,
            I32EnumAttrCase<"MatMulTopK", 4>
        ]>{
    let genSpecializedAttr = 0;
    let cppNamespace = "::tpu_mlir::tpu";
}
def Tpu_DistributionPatternAttr : EnumAttr<Tpu_Dialect, Tpu_DistributionPattern, "distribution_pattern">;


def Tpu_LutBF16Mode : I32EnumAttr<"LutBF16Mode",
        "bf16 look up table mode",
        [
            I32EnumAttrCase<"Other", 0>,
            I32EnumAttrCase<"Mantissa", 1>,
            I32EnumAttrCase<"Slope", 2>,
            I32EnumAttrCase<"Log", 3>,
            I32EnumAttrCase<"Exp", 4>,
        ]>{
    let genSpecializedAttr = 0;
    let cppNamespace = "::tpu_mlir::tpu";
}
def Tpu_LutBF16ModeAttr : EnumAttr<Tpu_Dialect, Tpu_LutBF16Mode, "lut_mode">;

def Tpu_ActiveMode : I32EnumAttr<"ActiveMode",
        "Activation mode for ActiveOp, for sigmoid/exp, e.g.",
        [
            I32EnumAttrCase<"TANH", 0>,
            I32EnumAttrCase<"SIGMOID", 1>,
            I32EnumAttrCase<"RELU", 2>,
            I32EnumAttrCase<"EXP", 3>,
            I32EnumAttrCase<"ELU", 4>,
            I32EnumAttrCase<"SQRT", 5>,
            I32EnumAttrCase<"SQUARE", 6>,
            I32EnumAttrCase<"RSQRT", 7>,
            I32EnumAttrCase<"ABSVAL", 8>,
            I32EnumAttrCase<"LN", 9>,
            I32EnumAttrCase<"ROUND", 10>,
            I32EnumAttrCase<"CEIL", 11>,
            I32EnumAttrCase<"FLOOR", 12>,
            I32EnumAttrCase<"SIN", 13>,
            I32EnumAttrCase<"COS", 14>,
            I32EnumAttrCase<"IS_FINITE", 15>,
            I32EnumAttrCase<"MISH", 16>,
            I32EnumAttrCase<"SWISH", 17>,
            I32EnumAttrCase<"HSWISH", 18>,
            I32EnumAttrCase<"SILU", 19>,
            I32EnumAttrCase<"ARCSIN", 20>,
            I32EnumAttrCase<"ARCCOS", 21>,
            I32EnumAttrCase<"ARCSINH", 22>,
            I32EnumAttrCase<"ARCCOSH", 23>,
            I32EnumAttrCase<"ARCTANH", 24>,
            I32EnumAttrCase<"SINH", 25>,
            I32EnumAttrCase<"COSH", 26>,
            I32EnumAttrCase<"TAN", 27>,
            I32EnumAttrCase<"SIGN", 28>,
            I32EnumAttrCase<"GELU", 29>,
            I32EnumAttrCase<"ERF", 30>,
            I32EnumAttrCase<"HSIGMOID", 31>,
            I32EnumAttrCase<"LOG_SIGMOID", 32>,
            I32EnumAttrCase<"SOFT_PLUS", 33>,
            I32EnumAttrCase<"SOFT_SIGN", 34>,
        ]>{
    let genSpecializedAttr = 0;
    let cppNamespace = "::tpu_mlir::tpu";
}
def Tpu_ActiveModeAttr : EnumAttr<Tpu_Dialect, Tpu_ActiveMode, "active_mode">;

def Tpu_ResizeMode : I32EnumAttr<"ResizeMode",
        "Resize mode",
        [
            I32EnumAttrCase<"nearest", 0>,
            I32EnumAttrCase<"linear", 1>,
        ]>{
    let genSpecializedAttr = 0;
    let cppNamespace = "::tpu_mlir::tpu";
}
def Tpu_ResizeModeAttr : EnumAttr<Tpu_Dialect, Tpu_ResizeMode, "mode">;

def Tpu_ResizeCoordMode : I32EnumAttr<"ResizeCoordMode",
        "Resize coord mode",
        [
            I32EnumAttrCase<"align_corners", 0>,
            I32EnumAttrCase<"half_pixel", 1>,
            I32EnumAttrCase<"pytorch_half_pixel", 2>,
            I32EnumAttrCase<"asymmetric", 3>,
        ]>{
    let genSpecializedAttr = 0;
    let cppNamespace = "::tpu_mlir::tpu";
}
def Tpu_ResizeCoordModeAttr : EnumAttr<Tpu_Dialect, Tpu_ResizeCoordMode, "coord_mode">;

def Tpu_RunMode: I32EnumAttr<"RunMode",
        "tpu dialect run mode for each subnet",[
            I32EnumAttrCase<"TPU_STATIC",  0>,
            I32EnumAttrCase<"TPU_DYNAMIC", 1>,
            I32EnumAttrCase<"CPU",         2>,
            I32EnumAttrCase<"SWITCH",      3>,
            I32EnumAttrCase<"MERGE",       4>,
            I32EnumAttrCase<"LOOP",       5>,
            I32EnumAttrCase<"UNKNOW",       6>
        ]> {
    let genSpecializedAttr = 0;
    let cppNamespace = "::tpu_mlir::tpu";
}
def Tpu_RunModeAttr : EnumAttr<Tpu_Dialect, Tpu_RunMode, "run_mode">;

//===----------------------------------------------------------------------===//
// Tpu Types.
//===----------------------------------------------------------------------===//

def AnyTensorOrNone: AnyTypeOf<[AnyRankedTensor, NoneType]>;

//===----------------------------------------------------------------------===//
// Tpu Operations.
//===----------------------------------------------------------------------===//
class Tpu_BaseOp<string mnemonic, list<Trait> traits = []> :
        Op<Tpu_Dialect, mnemonic, !listconcat(traits, [TpuTypeRestrict])>;

class Tpu_Op<string mnemonic, list<Trait> traits = []>  :
        Op<Tpu_Dialect, mnemonic, !listconcat(traits,
           [TpuTypeRestrict,
            DeclareOpInterfaceMethods<GlobalGenInterface>,
            DeclareOpInterfaceMethods<InferenceInterface>,
            DeclareOpInterfaceMethods<DynGlobalGenInterface>])>;

def Tpu_BufferOp : Tpu_BaseOp<"Buffer"> {
    let summary = "buffer operator";

    let description = [{
        A global buffer for operation, and free after op
    }];

    let results = (outs AnyRankedTensor:$output);
    let extraClassDeclaration = [{
        static mlir::Value create(mlir::Operation * OwnerOp,
                                  mlir::RankedTensorType& type);
    }];
}

def Tpu_Conv2DOp : Tpu_Op<"Conv2D", [SupportFuseRelu,
        DeclareOpInterfaceMethods<TypeInterface>,
        DeclareOpInterfaceMethods<IndexingMapsInterface>,
        DeclareOpInterfaceMethods<LocalGenInterface,
            ["BackwardH", "BackwardW", "LocalGenSupport", "assign_sec_info"]>,
        DeclareOpInterfaceMethods<DynLocalGenInterface,
            ["DynBackwardH", "DynBackwardKh", "DynBackwardStrideH", "DynBackwardUpPadH", "DynBackwardDownPadH", "DynForwardHeight"]>]> {
    let summary = "convolution 2d operator";

    let description = [{
    }];

    let arguments = (ins
            AnyRankedTensor:$input,
            AnyRankedTensor:$filter,
            AnyTensorOrNone:$bias,
            I64ArrayAttr:$kernel_shape,
            I64ArrayAttr:$strides,
            I64ArrayAttr:$pads,
            DefaultValuedAttr<I64Attr, "1">:$group,
            OptionalAttr<I64ArrayAttr>:$dilations,
            OptionalAttr<I64ArrayAttr>:$inserts,
            DefaultValuedAttr<BoolAttr, "false">:$do_relu,
            DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit,
            BoolAttr:$with_bias,
            DefaultValuedAttr<BoolAttr, "false">:$coeff_merged,
            DefaultValuedAttr<I64Attr, "0">:$use_3ic_optimize,
            DefaultValuedAttr<I64Attr, "0">:$kernel_zp,
            OptionalAttr<I64ArrayAttr>:$multiplier,
            OptionalAttr<I64ArrayAttr>:$rshift,
            DefaultValuedAttr<Tpu_RequantModeAttr, "tpu::RequantMode::MultiplierShift">:$quant_mode,
            OptionalAttr<Tpu_LayerGroupAttr>:$ginfo,
            OptionalAttr<BoolAttr>:$do_leaky_relu,
            OptionalAttr<F64Attr>:$neg_slope,
            OptionalAttr<SI32Attr>:$multiplier_pos,
            OptionalAttr<SI32Attr>:$multiplier_neg,
            OptionalAttr<SI32Attr>:$rshift_pos,
            OptionalAttr<SI32Attr>:$rshift_neg
            );

    let results = (outs AnyRankedTensor:$output);
    let extraClassDeclaration = [{
        conv_attr_t parseParam();
        void assign_fw_param(void *param);
    }];
}

def Tpu_Conv3DOp: Tpu_Op<"Conv3D", [SupportFuseRelu,
        DeclareOpInterfaceMethods<TypeInterface>,
        DeclareOpInterfaceMethods<LocalGenInterface,
            ["LocalGenSupport", "BackwardH", "BackwardW", "BackwardD", "assign_sec_info"]>,
        DeclareOpInterfaceMethods<DynLocalGenInterface,
            ["DynBackwardH", "DynBackwardKh", "DynBackwardStrideH", "DynBackwardUpPadH", "DynBackwardDownPadH", "DynForwardHeight"]>]> {
    let summary = "convolution 2d operator";

    let description = [{}];

    let arguments = (ins
        AnyRankedTensor:$input,
        AnyRankedTensor:$filter,
        AnyTensorOrNone:$bias,
        I64ArrayAttr:$kernel_shape,
        I64ArrayAttr:$strides,
        I64ArrayAttr:$pads, // front,top,left,back,bottom,right
        DefaultValuedAttr<I64Attr, "1">:$group,
        OptionalAttr<I64ArrayAttr>:$dilations,
        OptionalAttr<I64ArrayAttr>:$inserts,
        DefaultValuedAttr<BoolAttr, "false">:$do_relu,
        DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit,
        //new param
        BoolAttr:$with_bias,
        DefaultValuedAttr<I64Attr, "0">:$kernel_zp,
        OptionalAttr<I64ArrayAttr>:$multiplier,
        OptionalAttr<I64ArrayAttr>:$rshift,
        DefaultValuedAttr<Tpu_RequantModeAttr, "tpu::RequantMode::MultiplierShift">:$quant_mode,
        OptionalAttr<Tpu_LayerGroupAttr>:$ginfo
    );

    let results = (outs AnyRankedTensor:$output);
    let extraClassDeclaration = [{
        conv_attr_t parseParam();
        void assign_fw_param(void *param);
    }];
}

def Tpu_GroupOp : Tpu_BaseOp<"Group"> {
    let summary = "Group operation";
    let description = [{
        Make ops in one group to inferece by local mem
    }];
    let arguments = (ins
        Variadic<AnyRankedTensor>:$inputs,
        I64Attr:$nsecs,
        I64Attr:$hsecs,
        I64Attr:$dsecs,
        I64Attr:$wsecs,
        I64Attr:$csecs,
        I64Attr:$swpipl_stage_num,
        I64Attr:$group_type,
        // store timestep_idx(negative) and op_id(positive)
        DefaultValuedAttr<I64ArrayAttr, "{0}">:$flow,
        // only store op_id
        DefaultValuedAttr<I64ArrayAttr, "{}">:$self_up_overlap_op,
        DefaultValuedAttr<I64ArrayAttr, "{}">:$self_down_overlap_op,
        // store timestep_idx(negative) and op_id(positive)
        DefaultValuedAttr<I64ArrayAttr, "{}">:$other_up_overlap_op,
        DefaultValuedAttr<I64ArrayAttr, "{}">:$other_down_overlap_op
    );
    let results = (outs Variadic<AnyRankedTensor>:$outputs);
    let regions = (region SizedRegion<1>:$body);
}

def Tpu_ParallelOp : Tpu_BaseOp<"Parallel"> {
    let summary = "Parallel execution region";
    let description = [{
        The ops in one parallel should run in parallel.
    }];
    let arguments = (ins
        Variadic<AnyTensorOrNone>:$inputs
    );
    let results = (outs Variadic<AnyRankedTensor>:$outputs);
    let regions = (region SizedRegion<1>:$body);
}

def Tpu_CastOp : Tpu_Op<"Cast", [
        InOutSameShape, SupportElementwise,
        DeclareOpInterfaceMethods<LocalGenInterface, ["LocalGenSupport"]>,
        DeclareOpInterfaceMethods<DynLocalGenInterface>,
        DeclareOpInterfaceMethods<TypeInterface>]> {
    let summary = "Cast operation";
    let description = [{
    }];
    let arguments = (ins
        AnyRankedTensor:$input,
        OptionalAttr<BoolAttr>:$extra_input,
        OptionalAttr<Tpu_LayerGroupAttr>:$ginfo,
        DefaultValuedAttr<BoolAttr, "true">:$with_scale //for BM1684
    );
    let results = (outs AnyRankedTensor:$output);
    let hasCanonicalizer = 1;
    let extraClassDeclaration = [{
        void assign_fw_param(void *param);
    }];
}

def Tpu_YieldOp : Tpu_BaseOp<"Yield", [Terminator, HasParent<"GroupOp, ParallelOp, IfOp, LoopOp">]> {
    let summary = "Yield values to parent operation";
    let description = [{
    }];

    let arguments = (ins Variadic<AnyTensor>:$operands);

    let builders = [
        OpBuilder<(ins), [{ build($_builder, $_state, std::nullopt); }]>
    ];

}

def Tpu_IfOp : Tpu_Op<"If", [DeclareOpInterfaceMethods<RegionBranchOpInterface>,
                RecursiveMemoryEffects,
                NoRegionArguments]> {
    let summary = "if operation";
    let description = [{
        If conditional
    }];
    let arguments = (ins AnyTensor:$cond);
    let results = (outs Variadic<AnyRankedTensor>:$output);
    let regions = (region SizedRegion<1>:$then_branch,
                SizedRegion<1>:$else_branch);
    let extraClassDeclaration = [{
        static int getNumberOfOperands() {
            return 1;
        }
        static int getNumberOfResults() {
            return -1;
        }
        static std::vector<int> getTypeMap() {
            return {-1};
        }
        int64_t getSubgraphRegionIdx(const std::string& name) {
            if (name == "then_branch") return 0;
            if (name == "else_branch") return 1;
            llvm_unreachable("region with the specified name does not exist");
        }
    }];
}

def Tpu_LoopOp : Tpu_Op<"Loop", [DeclareOpInterfaceMethods<RegionBranchOpInterface>,
                    SingleBlockImplicitTerminator<"tpu::YieldOp">,
                    RecursiveMemoryEffects]> {
    let summary = "Loop operation";
    let description = [{
        Generic Looping construct, support while/do_while/for/forerver etc:
    }];

    let arguments = (ins AnyTypeOf<[AnyTensor, NoneType]>:$M,
                    AnyTypeOf<[AnyTensor, NoneType]>:$cond,
                    Variadic<AnyTypeOf<[AnyTensor, NoneType]>>:$v_initial);
    let results = (outs Variadic<AnyTypeOf<[AnyTensor, NoneType]>>:$v_final_and_scan_outputs);
    let regions = (region SizedRegion<1>:$body);
    let hasCanonicalizer = 1;
    let extraClassDeclaration = [{
        static int getNumberOfOperands() {
            return -1;
        }
        static int getNumberOfResults() {
            return -1;
        }
        static std::vector<int> getTypeMap() {
            return {22};
        }

        mlir::Operation::result_range v_final();
        mlir::Operation::result_range scan_outputs();
        int64_t getSubgraphRegionIdx(const std::string& name) {
            if (name == "body") return 0;
            llvm_unreachable("region with the specified name does not exist");
        }
    }];
}
#endif //TPU_MLIR_TPU_OPS