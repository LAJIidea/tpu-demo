//
// Created by kingkiller on 2023/9/6.
//

#pragma once

typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned char u8;
typedef signed char s8;
#define MAX_CONCAT_INPUT_NUM 16
#define MAX_ELET_INPUT_NUM 10
#define MAX_SPLIT_OUTPUT_NUM 8
#define MAX_SHAPE_DIMS 8
#define MAX_YOLO_INPUT_NUM 8
#define MAX_YOLO_ANCHOR_NUM 8
typedef int LayerId;

typedef struct fw_conv_layer_param {
    u32 ic_oc;
    u32 concat_c; // full channels for local shape
    u32 groups;
    u32 kh_kw;
    u8 dh;
    u8 dw;
    u8 pad_h;
    u8 pad_h_after;
    u8 pad_w;
    u8 pad_w_after;
    u8 stride_h;
    u8 stride_w;
    u8 using_bias;
    u8 if_relu;
    float relu_upper_limit;
    u8 use_winograd;
    u32 c_idx;        // for local concat useage
    u32 reference_id; // for local concat , reference_id is the true ouput tensor
    // of conv
    u8 rshiftbits;
    u8 opd0_sign;
    u8 opd1_sign;
    u8 opd2_sign;
    u8 res_sign;
    u8 mulshift;
    int mulvalue;
    int mulshiftnum;
    int weight_is_tensor;
    u32 scale_dim[4];
    int scale_axis;
    int scale_axis_num;
    u32 scale_bias;
    u32 if_batchnorm;
    u32 if_scale;
    u32 if_double_buffer;
    u64 weight_global_offset;
    u32 double_buffer_local_offset;
    u32 is_tf_same_pad;
} fw_conv_layer_param_t;
