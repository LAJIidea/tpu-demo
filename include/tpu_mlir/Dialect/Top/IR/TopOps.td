#ifndef TPU_MLIR_TOP_OPS
#define TPU_MLIR_TOP_OPS

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "tpu_mlir/Interfaces/InferenceInterface.td"
include "tpu_mlir/Interfaces/FlopsInterface.td"
include "tpu_mlir/Interfaces/ShapeInterface.td"
include "tpu_mlir/Traits/Traits.td"

//==========================
//
// Defines TOP Dialect.
//
//==========================

def Top_Dialect : Dialect {
  let name = "top";
  let summary = "A topdialect for the TPU_MLIR specification";
  let cppNamespace = "::tpu_mlir::top";
}

//===--------------------------------===//
// TOP Attributes.
//===--------------------------------===//

class Top_Attr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Top_Dialect, attrName, traits> {
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
    !foldl(!head(cases), !tail(cases),
          prev, cur, prev # ", or " # cur)>;
def ArgModeAttr: AnyStrAttrOf<["ArgMin","ArgMax"]>;
def CompareModeAttr: AnyStrAttrOf<["Equal","Greater","GreaterOrEqual","Less","LessOrEqual", "NotEqual", "And", "Not"]>;
def ReduceModeAttr: AnyStrAttrOf<["ReduceMin","ReduceMax","ReduceMean","ReduceL2","ReduceL1","ReduceSum","ReduceProd"]>;
def InterpModeAttr: AnyStrAttrOf<["nearest","linear"]>;
def InterpCoordModeAttr: AnyStrAttrOf<["align_corners", "half_pixel", "pytorch_half_pixel", "asymmetric"]>;
def PixelFormatAttr: AnyStrAttrOf<["rgb","bgr","gray","rgba"]>;
def ChannelFormatAttr: AnyStrAttrOf<["nhwc","nchw"]>;
def PadModeAttr: AnyStrAttrOf<["normal","center"]>;
def DetectionOutputCodeTypeAttr: AnyStrAttrOf<["CORNER", "CENTER_SIZE", "CORNER_SIZE"]>;
def RoiAlignModeAttr: AnyStrAttrOf<["Avg","Max"]>;
def NonZeroOrderAttr: AnyStrAttrOf<["ColMajor","RowMajor"]>;
def StoreModeAttr: AnyStrAttrOf<["1N", "2N", "4N"]>;
def YoloVersionAttr: AnyStrAttrOf<["yolov3", "yolov3_tiny", "yolov3_spp", "yolov4", "yolov5"]>;
def MatchTemplateModeAttr: AnyStrAttrOf<["TM_CCOEFF_NORMED", "TM_SQDIFF"]>;
def PaddingModeAttr:AnyStrAttrOf<["constant","reflect","symmetric","edge"]>;

//===----------------------------===//
// TOP Types.
//===----------------------------===//

def AnyTensorOrNone: AnyTypeOf<[AnyTensor, NoneType]>;

//===----------------------------===//
// TOP Op Definition.
//===----------------------------===//

// ==== BaseOp ====
class Top_BaseOp<string mnemonic, list<Trait> traits = []> :
    Op<Top_Dialect, mnemonic, traits> ;

def Top_NoneOp : Top_BaseOp<"None"> {
    let summary = "none operator";

    let description = [{
        A none Op to return a NoneType.
    }];
    let results = (outs NoneType);
}

def Top_WeightOp : Top_BaseOp<"Weight"> {
    let summary = "load weight operator";

    let description = [{
        Load weight from a file. The file should be a valid .npz format file.
        This Op does not take any input, and the location captures the tensor name.
        The Output is an n-dimensional tensor whose type matches
                the tensor type in the .npz file.
    }];

    let arguments = (ins
        OptionalAttr<F64ArrayAttr>:$scale,
        OptionalAttr<BoolAttr>:$do_compress,
        OptionalAttr<StoreModeAttr>:$store_mode,
        OptionalAttr<I64ArrayAttr>:$allow_split
    );

    let results = (outs AnyRankedTensor:$output);
    let extraClassDeclaration = [{
        template<typename T>
        std::shared_ptr<std::vector<T>> read();
        std::shared_ptr<std::vector<float>> read_as_float();
        std::shared_ptr<std::vector<int32_t>> read_as_int32();
        std::shared_ptr<std::vector<uint8_t>> read_as_byte();
        template<typename T>
        static mlir::Value create(mlir::Operation * OwnerOp,
                                  llvm::StringRef suffix,
                                  const std::vector<T>& data,
                                  mlir::RankedTensorType& type,
                                  uint32_t store_mode = 0);
        template<typename T>
        mlir::LogicalResult update(const std::vector<T>& data, size_t count);
        mlir::Value clone_bf16(mlir::Operation * OwnerOp, std::string name = "");
        mlir::Value clone_f16(mlir::Operation * OwnerOp);
        mlir::Value clone_int(mlir::Operation *OwnerOp);
        mlir::Value clone(llvm::StringRef suffix);
    }];
}

def Top_InputOp : Top_BaseOp<"Input"> {
    let summary = "Input operator";

    let description = [{
    }];

    let arguments = (
      ins AnyRankedTensor:$input,
      // preprocess for input
      OptionalAttr<PixelFormatAttr>:$pixel_format,
      OptionalAttr<ChannelFormatAttr>:$channel_format,
      OptionalAttr<I64ArrayAttr>:$resize_dims,
      OptionalAttr<BoolAttr>:$keep_aspect_ratio,
      OptionalAttr<StrAttr>:$keep_ratio_mode,
      OptionalAttr<I64Attr>:$pad_value,
      OptionalAttr<PadModeAttr>:$pad_type,
      OptionalAttr<F64ArrayAttr>:$scale,
      OptionalAttr<F64ArrayAttr>:$mean,
      // for cv18xx fusepreprocess
      OptionalAttr<StrAttr>:$customization_format,
      OptionalAttr<BoolAttr>:$aligned
            );

    let results = (outs AnyTensor:$output);
}

def Top_TupleOp : Top_BaseOp<"Tuple"> {
    let summary = "Tuple operator";
    let description = [{
        gen by torch prim::TupleConstruct, y = (a, b)
    }];

    let arguments = (ins
      Variadic<AnyTensor>:$inputs);

    let results = (outs AnyTensor:$output);
}

def Top_UnTupleOp : Top_BaseOp<"UnTuple"> {
    let summary = "UnTuple operator";
    let description = [{
        gen by torch prim::TupleUnpack, a, b = y
    }];

    let arguments = (ins
      Variadic<AnyTensor>:$inputs);

    let results = (outs
      Variadic<AnyTensor>:$outputs);
}

// ==== Function Op ====
class Top_Op<string mnemonic, list<Trait> traits = []> :
    Top_BaseOp<mnemonic, !listconcat(traits,
        [DeclareOpInterfaceMethods<InferenceInterface>,
         DeclareOpInterfaceMethods<FlopsInterface>,
         DeclareOpInterfaceMethods<ShapeInterface>])>;

def Top_BatchNormOp : Top_Op<"BatchNorm", [SupportFuseRelu]> {
    let summary = "BatchNormalization operator";
    let description = [{
        Applies Batch Normalization over a 4D input (a mini-batch of 2D inputs
        with additional channel dimension) as described in the paper
        Batch Normalization: Accelerating Deep Nerwork Training by Reducing
        Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`___ .

        ```math
            y = \frac{x - \mathrm{E}[x]}{ \sqrt{\matrm{Var}[x] + \epsilon}} * \gamma + \beta
        ```

        The mean and standard-deviation are calculated per-dimension over
        the mini-batches and $$\gamma$$ and $$\beta$$ are learnable parameter vectors
        of size C (where C is the input channel size).
    }];
    let arguments = (ins
      AnyTensor:$input,
      AnyTensor:$mean,
      AnyTensor:$variance,
      AnyTensorOrNone:$gamma,
      AnyTensorOrNone:$beta,
      DefaultValuedAttr<F64Attr, "1e-05">:$epsilon,
      DefaultValuedAttr<BoolAttr, "false">:$do_relu,
      DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit);
    let results = (outs AnyTensor:$output);
    let hasCanonicalizer = 1;
}

def Top_ConcatOp : Top_Op<"Concat", [SupportFuseRelu]> {
    let summary = "Concat operator";

    let description = [{
        Concatenates the given sequence of seq tensors in the given dimension.
        All tensors must either have the same shape (except in the concatenating dimension) or be empty.
    }];

    let arguments = (
        ins Variadic<AnyTensor>:$inputs,
        DefaultValuedAttr<SI32Attr, "1">:$axis,
        DefaultValuedAttr<BoolAttr, "false">:$do_relu,
        DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit
    );

    let results = (outs AnyTensor:$output);
    let hasCanonicalizer = 1;
}

def Top_ConvOp: Top_Op<"Conv", [SupportFuseRelu,
        DeclareOpInterfaceMethods<InferenceInterface,["backward_weight"]>]> {
    let summary = "Convolution operator";

    let description = [{
        In the simplest case, the output value of the layer with input size
                $$(N, C_{\text{in}}, H, W)$$ and output $$(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})$$
                can be precisely described as:

        ```math
        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{input}(N_i, k)
        ```


        where $$\star$$ is the valid 2D cross-correlation operator,
        $$N$$ is a batch size, $$C$$ denotes a number of channels,
                $$H$$ is a height of input planes in pixels, and $$W$$ is
                width in pixels.

        weight (Tensor): the learnable weights of the module of shape
        $$(\text{out\_channels}, \frac{\text{in\_channels}}{\text{groups}},
        \text{kernel\_size[0]}, \text{kernel\_size[1]})$$

                bias (Tensor optional): the learnable bias of the module of shape (out_channels).

        - **stride**: controls the stride for the cross-correlation, a single
        number or a tuple.

        - **padding**: controls the amount of padding applied to the input. It
                contains four ints with top, left, bottom, right respectively.

        - **dilation**: controls the spacing between the kernel points; also
                known as the à trous algorithm. It is harder to describe, but this
        [Link](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md)
        has a nice visualization of what **dilation** does.

        - **groups**: (optional): Number of blocked connections from input
        channels to output channels. Default: 1


        Shape:
        - Input: $$(N, C_{in}, H_{in}, W_{in})$$
        - Output: $$(N, C_{out}, H_{out}, W_{out})$$ where

        ```math
                H_{out} = \left\lfloor\frac{H_{in}  + \text{padding}[0] + \text{padding}[2] - \text{dilation}[0]
        \times (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor
        ```
        ```math
                W_{out} = \left\lfloor\frac{W_{in}  + \text{padding}[1] + \text{padding}[3] - \text{dilation}[1]
        \times (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor
        ```
    }];

    let arguments = (ins
        AnyTensor:$input,
        AnyTensor:$filter,
        AnyTensorOrNone:$bias,
        I64ArrayAttr:$kernel_shape,
        I64ArrayAttr:$strides,
        I64ArrayAttr:$pads, // top,left,bottom,right
        DefaultValuedAttr<I64Attr, "1">:$group,
        OptionalAttr<I64ArrayAttr>:$dilations,
        OptionalAttr<I64ArrayAttr>:$inserts,
        DefaultValuedAttr<BoolAttr, "false">:$do_relu,
        DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit,
        OptionalAttr<F64Attr>:$in_int4_scale,
        OptionalAttr<F64Attr>:$in_int4_zp,
        OptionalAttr<F64Attr>:$out_int8_scale,
        OptionalAttr<F64Attr>:$out_int8_zp
    );

    let results = (outs AnyTensor:$output);
    let hasCanonicalizer = 1;
    let extraClassDeclaration = [{
        conv_attr_t parseParam();
    }];
}

def Top_DivOp : Top_Op<"Div"> {
    let summary = "Div operator";

    let description = [{
        Performs element-wise binary division.
    }];

    let arguments = (ins
            Variadic<AnyTensor>:$inputs,
            DefaultValuedAttr<BoolAttr, "false">:$is_reverse,
            DefaultValuedAttr<BoolAttr, "false">:$do_relu,
            DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit
            );

    let results = (outs AnyTensor:$output);
    let hasCanonicalizer = 1;
}

def Top_ScaleOp : Top_Op<"Scale", [SupportFuseRelu]> {
    let summary = "Scale operator";

    let description = [{
        Y = X * S + B,
        where the shape of X/Y is [n, c, h, w] and the shape of S/B is [1, c, 1, 1].
    }];

    let arguments = (ins
      AnyTensor:$input,
      AnyTensor:$scale,
      AnyTensor:$bias,

      DefaultValuedAttr<BoolAttr, "false">:$do_relu,
      DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit);

    let results = (outs AnyTensor:$output);
    let hasCanonicalizer = 1;
}

def Top_Depth2SpaceOp : Top_Op<"Depth2Space"> {
    let summary = "Depth2Space operator";

    let description = [{
        Refer to `https://github.com/onnx/onnx/blob/main/docs/Operators.md#depthtospace`
        [n, c, h, w] => [n, c / (block_h * block_w), h * block_h, w * block_w];
        if inversed, [n, c, h, w] => [n, c * block_h * block_w, h / block_h, w / block_w];

        if DCR(depth-column-row), channel ordered by block_h * block_w * c;
        else CRD(column-row-depth), channel ordered by c * block_h * block_w;

        The format of input or output is NCHW or NHWC.
    }];

    let arguments = (
      ins AnyTensor:$input,
      I64Attr:$block_h,
      I64Attr:$block_w,
      BoolAttr:$is_CRD,
      BoolAttr:$is_inversed,
      DefaultValuedAttr<BoolAttr, "true">:$in_is_NCHW,
      DefaultValuedAttr<BoolAttr, "true">:$out_is_NCHW,
      DefaultValuedAttr<BoolAttr, "false">:$swap_cr
            );

    let results = (outs AnyTensor:$output);
    let hasCanonicalizer = 1;
}

def Top_AddOp : Top_Op<"Add", [SupportFuseRelu]> {
    let summary = "add operator";

    let description = [{
        Elementwise addition of input1 and input2. Axis of size 1 will be broadcast,
        as necessary.
    }];

    let arguments = (ins
            Variadic<AnyTensor>:$inputs,
            DefaultValuedAttr<BoolAttr, "false">:$do_relu,
            DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit,
            OptionalAttr<F64ArrayAttr>:$coeff
            );

    let results = (outs AnyTensor:$output);
    let hasCanonicalizer = 1;
}

def Top_SubOp: Top_Op<"Sub", [SupportFuseRelu]> {
    let summary = "sub operator";

    let description = [{
        Elementwise subtraction of input1 and input2. Axis of size 1 will be broadcast,
        as necessary.
    }];

    let arguments = (ins
        Variadic<AnyTensor>:$inputs,
        DefaultValuedAttr<BoolAttr, "false">:$is_reverse,
        DefaultValuedAttr<BoolAttr, "false">:$do_relu,
        DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit,
        OptionalAttr<F64ArrayAttr>:$coeff
    );

    let results = (outs AnyTensor:$output);
}

def Top_MulOp: Top_Op<"Mul", [SupportFuseRelu]> {
    let summary = "Mul operator";

    let description = [{
        Elementwise multiplication of input1 and input2. input1 and input2 are tensors.
    }];

    let arguments = (ins
        Variadic<AnyTensor>:$inputs,
        DefaultValuedAttr<BoolAttr, "false">:$do_relu,
        DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit
    );

    let results = (outs AnyTensor:$output);
    let hasCanonicalizer = 1;
}

def Top_MinOp: Top_Op<"Min"> {
    let summary = "min operator";

    let description = [{
        Element-wise min of each of the input tensors. All inputs and outputs must have the same data type.
    }];

    let arguments = (ins
        Variadic<AnyTensor>:$inputs
    );

    let results = (outs AnyTensor:$output);
    let hasCanonicalizer = 1;
}

def Top_MaxOp: Top_Op<"Max"> {
    let summary = "max operator";

    let description = [{
        Element-wise max of each of the input tensors. All inputs and outputs must have the same data type.
    }];

    let arguments = (ins
        Variadic<AnyTensor>:$inputs
    );

    let results = (outs AnyTensor:$output);
    let hasCanonicalizer = 1;
}

def Top_AddConstOp: Top_Op<"AddConst", [SupportFuseRelu, SupportPermuteMove]> {
    let summary = "Add Const operator";

    let description = [{
        Y = X + const_val
    }];

    let arguments = (ins
        AnyTensor:$input,
        F64Attr: $const_val,
        DefaultValuedAttr<BoolAttr, "false">:$do_relu,
        DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit
    );

    let results = (outs AnyTensor:$output);
}

def Top_SubConstOp: Top_Op<"SubConst", [SupportFuseRelu]> {
    let summary = "Sub Const operator";

    let description = [{
        Y = X - const_val or const_val - X
    }];

    let arguments = (ins
        AnyTensor:$input,
        F64Attr: $const_val,
        DefaultValuedAttr<BoolAttr, "false">:$is_reverse,
        DefaultValuedAttr<BoolAttr, "false">:$do_relu,
        DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit
    );

    let results = (outs AnyTensor:$output);
}

def Top_MulConstOp: Top_Op<"MulConst", [SupportFuseRelu, SupportPermuteMove]> {
    let summary = "Mul Const operator";

    let description = [{
        Y = X * const_val
    }];

    let arguments = (ins
        AnyTensor:$input,
        F64Attr: $const_val,
        DefaultValuedAttr<BoolAttr, "false">:$do_relu,
        DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit
    );

    let results = (outs AnyTensor:$output);
    let hasCanonicalizer = 1;
}

def Top_MinConstOp: Top_Op<"MinConst"> {
    let summary = "Min Const operator";

    let description = [{
        Y = Min(X, const_val)
    }];

    let arguments = (ins
        AnyTensor:$input,
        F64Attr: $const_val
    );

    let results = (outs AnyTensor:$output);
}

def Top_MaxConstOp: Top_Op<"MaxConst"> {
    let summary = "Max Const operator";

    let description = [{
        Y = Max(X, const_val)
    }];

    let arguments = (ins
        AnyTensor:$input,
        F64Attr: $const_val
    );

    let results = (outs AnyTensor:$output);
}

def Top_MatMulOp : Top_Op<"MatMul", [SupportFuseRelu]> {
    let summary = "matmul operator";

    let description = [{
        Performs a two dimensional matrix multiplication. This allows both inputs to
        be activations, rather than reserving weights as an attribute in the
        FULLY_CONNECTED operator.
    }];

    let arguments = (ins
            AnyTensor:$input,
            AnyTensor:$right,
            AnyTensorOrNone:$bias,
            DefaultValuedAttr<BoolAttr, "false">:$right_transpose,
            DefaultValuedAttr<BoolAttr, "false">:$left_transpose,
            DefaultValuedAttr<BoolAttr, "false">:$output_transpose,
            DefaultValuedAttr<BoolAttr, "false">:$hdim_is_batch,
            DefaultValuedAttr<BoolAttr, "true">:$keep_dims,
            DefaultValuedAttr<BoolAttr, "false">:$do_relu,
            DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit,
            OptionalAttr<F64Attr>:$in_int4_scale,
            OptionalAttr<F64Attr>:$in_int4_zp,
            OptionalAttr<F64Attr>:$out_int8_scale,
            OptionalAttr<F64Attr>:$out_int8_zp
            );

    let results = (outs AnyTensor:$output);
    let hasCanonicalizer = 1;
    let extraClassDeclaration = [{
        matmul_attr_t parseParam();
    }];
}

def Top_PermuteOp : Top_Op<"Permute"> {
    let summary = "Permute operator";

    let description = [{
        Perform permute on input.
    }];

    let arguments = (
      ins AnyTensor:$input,
      I64ArrayAttr:$order
            );
    let results = (outs AnyTensor:$output);
    let hasCanonicalizer = 1;
    let extraClassDeclaration = [{
        permute_attr_t parseParam();
    }];
}

def Top_ReluOp : Top_Op<"Relu", [SupportPermuteMove]> {
    let summary = "Relu operator";

    let description = [{
        ReLU with a scalar maximum value. if limit is zero, do not use upper limit.
    }];

    let arguments = (
      ins AnyTensor:$input,
      DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit
    );

    let results = (outs AnyTensor:$output);

    let hasCanonicalizer = 1;
}

def Top_YieldOp : Top_BaseOp<"Yield", [Terminator, HasParent<"IfOp, LoopOp">]> {
    let summary = "Yield operation";
    let description = [{
        The `top.Yield` operation represents a return operation within an subgraph.
        The operation takes variable number of operands and produces no results.

        This operation is not part of the standard and was added to assist tpu-mlr.
    }];

    let arguments = (ins Variadic<AnyTensor>:$operands);
}

def Top_IfOp : Top_Op<"If"> {
    let summary = "if operation";
    let hasVerifier = 1;
    let description = [{
        If conditional
    }];
    let arguments = (ins AnyTensor:$cond);
    let results = (outs Variadic<AnyTensor>:$output);
    let regions = (region SizedRegion<1>:$then_branch,
        SizedRegion<1>:$else_branch
        );
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

def Top_LoopOp : Top_Op<"Loop"> {
    let summary = "Loop operation";
    let description = [{
        Generic Looping construct. This loop has multiple termination conditions:

        1)  Trip count. Iteration count specified at runtime. Set by
            specifying the input M. Optional. Set to empty string to omit.
            Note that a static trip count (specified at graph construction time) can be
            specified by passing in a constant node for input M.
        2)  Loop termination condition. This is an input to the op that determines
            whether to run the first iteration and also a loop-carried dependency for
            the body graph. The body graph must yield a value for the condition variable,
            whether this input is provided or not.
        This table summarizes the operating modes of this operator with equivalent
        C-style code:

            Operator inputs defined as (max_trip_count, condition_var).

            input (\"\", \"\"):
                for (int i=0; ; ++i) {
                    cond = ... // Note this value is ignored, but is required in the body
                }

            input (\"\", cond) // Note this is analogous to a while loop
                bool cond = ...;
                for (int i=0; cond; ++i) {
                    cond = ...;
                }

            input (\"\", 1) // Note this is analogous to a do-while loop
                bool cond = true
                for (int i=0; cond; ++i) {
                    cond = ...;
                }

            input (trip_count, \"\") // Note this is analogous to a for loop
                int trip_count = ...
                for (int i=0; i < trip_count; ++i) {
                    cond = ...; // ignored
                }

            input (trip_count, cond)
                int trip_count = ...;
                bool cond = ...;
                for (int i=0; i < trip_count && cond; ++i) {
                    cond = ...;
                }


        *Sample usage - cond as well as trip count*

            graph predict-net {
                %a = Constant[value = <Scalar Tensor [3]>]()
                %b = Constant[value = <Scalar Tensor [6]>]()
                %keepgoing = Constant[value = <Scalar Tensor [1]>]()
                %max_trip_count = Constant[value = <Scalar Tensor [10]>]()
                %keepgoing_out, %b_out, %user_defined_vals = Loop[body = <graph body-net>](%max_trip_count, %keepgoing, %b)
                return
            }

            graph body-net (
                %i[INT32, scalar]           // iteration number
                %keepgoing_in[BOOL, scalar] // incoming loop-termination-condition; not used
                %b_in[INT32, scalar]        // incoming value of loop-carried-dependency b
            ) {
                %my_local = Add(%a, %b_in)
                %b_out = Sub(%a, %b_in) // outgoing value of loop-carried-dependency b
                %keepgoing_out = Greater(%my_local, %b_out) // outgoing loop-termination-condition
                %user_defined_val = Add(%b_in, %b_in) // scan-output value to be accumulated
                return %keepgoing_out, %b_out, %user_defined_val
            }

        *Sample equivalent C code*

            {
                /* User-defined code (enclosing scope) */
                int a = 3, b = 6;
                bool keepgoing = true; // Analogous to input cond
                /* End user-defined code */

                /* Implicitly-defined code */
                const int max_trip_count = 10; // Analogous to input M
                int user_defined_vals[]; // Imagine this is resizable
                /* End implicitly-defined code */
                /* initialize loop-carried variables and scan-output variables */
                bool keepgoing_out = keepgoing
                int b_out = b

                for (int i=0; i < max_trip_count && keepgoing_out; ++i) {
                /* Implicitly-defined code: bind actual parameter values
                   to formal parameter variables of loop-body */
                    bool keepgoing_in = keepgoing_out;
                    bool b_in = b_out;

                    /* User-defined code (loop body) */
                    int my_local = a + b_in; // Reading value \"a\" from the enclosing scope is fine
                    b_out = a - b_in;
                    keepgoing_out = my_local > b_out;
                    user_defined_val = b_in + b_in; // b_in and b_out are different variables
                    /* End user-defined code */

                    /* Implicitly defined-code */
                    user_defined_vals[i] = user_defined_val // accumulate scan-output values
                }
                // int t = my_local; // Can't do this. my_local is not accessible here.

                // The values below are bound to the output variables of the loop and therefore accessible
                // b_out; user_defined_vals; keepgoing_out;
            }

        There are several things of note in this code snippet:

        1)  Values from the enclosing scope (i.e. variable \"a\" here) are in scope and can
            be referenced in the inputs of the loop.
        2)  Any values computed in the loop body that needs to be used in a subsequent
            iteration or after the loop are modelled using a pair of variables in the loop-body,
            consisting of an input variable (eg., b_in) and an output variable (eg., b_out).
            These are referred to as loop-carried dependences. The loop operation node
            supplies the input value of the input variable for the first iteration, and
            returns the output value of the output variable produced by the final
            iteration.
        3)  Scan_output variables are used to implicitly concatenate values computed across
            all the iterations. In the above example, the value of user_defined_val computed
            over all iterations are concatenated and returned as the value of user_defined_vals
            after the loop.
        4)  Values created in the body cannot be accessed in the enclosing scope,
            except using the mechanism described above.

        Note that the semantics of this op support \"diagonal\" or \"wavefront\" execution.
        (See Step 3 here for an example:
        https://devblogs.nvidia.com/optimizing-recurrent-neural-networks-cudnn-5/).
        Frontends should emit multi-layer RNNs as a series of While operators (with
        time being the inner looping dimension), with each successive layer consuming
                the scan_outputs from the previous layer, possibly going through several
        point-wise operators (e.g. dropout, residual connections, linear layer).

        The input/output of subgraph (produced by loop node) matching is based on order instead of name. The implementation will figure out the names based on this order.
    }];
    let arguments = (ins AnyTypeOf<[AnyTensor, NoneType]>:$M,
        AnyTypeOf<[AnyTensor, NoneType]>:$cond,
        Variadic<AnyTypeOf<[AnyTensor, NoneType]>>:$v_initial
        );
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

def Top_ReshapeOp : Top_Op<"Reshape"> {
    let summary = "Reshape operation";
    let description = [{
        Returns a tensor with the same type/values as the input, with a new shape
        specified by the shape argument. Reshape may operate on tensors of any rank.
        No data conversion happens during a reshape operation.
        0: keep dim from input
        -1: left dim from input
    }];
    let arguments = (ins
      AnyTensor:$input,
      OptionalAttr<I64ArrayAttr>:$shape);
    let results = (outs AnyTensor:$output);
    let hasCanonicalizer = 1;
}
#endif // Top_OPS