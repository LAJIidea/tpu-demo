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
        the tensor type in .npz file.
    }];

    let arguments = (ins
      OptionalAttr<F64ArrayAttr>:$scale,
      OptionalAttr<BoolAttr>:$do_compress,
      OptionalAttr<StoreModeAttr>:$store_mode,
      OptionalAttr<I64ArrayAttr>:$allow_split);

    let results = (outs AnyRankedTensor:$output);
    let extraClassDeclaration =[{
    template<typename T>
    std::shared_ptr<std::vector<T>> read();
    std::shared_ptr<std::vector<float>> read_as_float();
    std::shared_ptr<std::vector<int32_t>> read_as_int32();
    std::shared_ptr<std::vector<uint8_t>> read_as_byte();
    template<typename T>
    mlir::LogicalResult update(const std::vector<T>& data, size_t count);
    mlir::Value clone_bf16(mlir::Operation * OwnerOp, std::string name = "");
    mlir::Value clone_f16(mlir::Operation * OwnerOp);
    mlir::Value clone_int(mlir::Operation *OwnerOp);
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