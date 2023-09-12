//
// Created by kingkiller on 2023/9/6.
//

#pragma once

#include "tpu_mlir/Backend/Arch.h"

#define MAX_CONV_IC (4095 - 32)
#define MAX_TIU_CHL (4095 - 32)
#define MAX_CHANNEL (4095 - 32)
#define MAX_HEIGHT (4095 - 32)
#define MAX_WIDTH (4095 - 32)
#define MAX_ROW (4095 - 32)
#define MAX_COL (4095 - 32)

typedef struct cvikernel_tensor_lmem_shape {
    uint32_t n, c, h, w;
} cvk_tl_shape_t;

typedef struct cvikernel_tensor_lmem_stride {
    uint32_t n, c, h, w;
} cvk_tl_stride_t;

typedef enum CVIKERNEL_FMT_E {
    CVK_FMT_F32 = 0,
    CVK_FMT_F16,
    CVK_FMT_I32,
    CVK_FMT_I16,
    CVK_FMT_I8,
    CVK_FMT_I4,
    CVK_FMT_I2,
    CVK_FMT_I1,
    CVK_FMT_U32,
    CVK_FMT_U16,
    CVK_FMT_U8,
    CVK_FMT_BF16,
    CVK_FMT_INVALID
} cvk_fmt_t;

typedef struct cvikernel_chip_info {
    uint32_t version;
    uint32_t node_num;
    uint32_t node_shift;
    uint32_t npu_num;
    uint32_t npu_shift;
    uint32_t eu_num;
    uint32_t eu_shift;
    uint32_t lmem_size;
    uint32_t lmem_shift;
    uint32_t lmem_banks;
    uint32_t lmem_bank_size;
    uint64_t lmem_start;
    uint64_t gmem_start;
    uint64_t gmem_size;
    uint64_t features;
} cvk_chip_info_t;

typedef struct cvikernel_tensor_lmem {
    uint32_t start_address;
    cvk_fmt_t fmt;
    cvk_fmt_t cmprs_fmt;
    cvk_tl_shape_t shape;
    cvk_tl_stride_t stride;
    uint8_t int8_rnd_mode; // 0 is round to nearset even, 1 is toward zero,
    // currently used by lut
    uint8_t eu_align;
} cvk_tl_t;

typedef struct {
    uint8_t mv_lut_idx;
    uint8_t mv_lut_base;
    const cvk_tl_t *src;
    const cvk_tl_t *dst;
    uint8_t outstanding; // Concurrent TDMA LD/ST and TDM L2L
    uint16_t layer_id;
} cvk_tdma_l2l_tensor_copy_param_t;

typedef struct {
    const cvk_tl_t *src;
    const cvk_tl_t *dst;
    int right_shift;
    uint32_t lrn_step;
    uint16_t layer_id;
} cvk_tdma_l2l_tensor_lrn_shift_param_t;

/*
 * Kernel Context
 */
typedef struct cvikernel_context {
    cvk_chip_info_t info;
//    cvk_operations_t *ops;
//    cvk_misc_operations_t *misc_ops;
    void *priv_data;
} cvk_context_t;

namespace tpu_mlir {
    namespace backend {
        class CV18xx : public Arch {
        public:
            static CV18xx &instance(module::Chip chip) {
                static CV18xx inst(chip);
                cv18xx = &inst;
                return inst;
            }

            enum GlobalMemoryRegion {
                SHARED_MEMORY = 0,
                WEIGHT_MEMORY = 1,
                PRIVATE_MEMORY = 2,
                IO_MEMORY_0 = 3,
                IO_MEMORY_1 = 4,
                IO_MEMORY_2 = 5,
                IO_MEMORY_3 = 6,
                IO_MEMORY_4 = 7,
                MAX_GLOBAL_MEMORY_REGION = 8
            };

            enum QuantizeMode {
                INT8_PER_LAYER = 1, // 1880 mode, scale + rightshift
                INT8_PER_CHANNEL = 2,
                INT8_32_MULTIPLER = 3, // 32bit multipliers(channel align) product tensor
                INT8_NOTSUPPORT        // not support, should be assert it
            };


        public:
            // ####################################################
            // backend common api
            // ####################################################

            //
            // shape/size/fmt functions
            //
            static void assert_support_fmt(cvk_fmt_t fmt);

            static int bitsize_of_fmt(uint32_t fmt);

            static int bytesize_of_fmt(cvk_fmt_t fmt) {
                return bitsize_of_fmt(fmt) / 8; // byte
            }

            static cvk_tl_shape_t tl_shape_t4(int n, int c, int h, int w) {
                return {static_cast<uint32_t>(n), static_cast<uint32_t>(c),
                        static_cast<uint32_t>(h), static_cast<uint32_t>(w)};
            }

            static int tensor_size(int n, int c, int h, int w, cvk_fmt_t fmt) {
                return n * c * h * w * bytesize_of_fmt(fmt);
            }

            static uint32_t tiu_eu_num(cvk_fmt_t fmt) {
                return EU_BYTES / (fmt == CVK_FMT_BF16 ? 2 : 1);
            }
            // tiu simple api
            static void tiu_zeros(uint16_t layer_id, cvk_tl_t *tl_mem);

            //
            // tiling functions
            //

            typedef enum TilingDim {
                TilingAll = 0, // reshape(NxCxHxW) and tiling to [1, NPU_NUM, x, EU_NUM]
                TilingNHW,     // keep c, tiling n,h,w
                TilingNCH,     // keep w, tiling n,c,h
                TilingNH,      // keep cw, tiling n,h
                TilingNCHW,    // tiling n,c,h,w
            } tiling_mode_t;

            typedef struct tiling_info {
                int32_t n;
                int32_t c;
                int32_t h;
                int32_t w;
                int32_t pos_n;
                int32_t pos_c;
                int32_t pos_h;
                int32_t pos_w;
                uint64_t offset; // gmem offset
            } tiling_info_t;


            static uint32_t chan_quan_param_size(bool do_bias) {
                // bias(4B) + multiplier(4B) + right_shift(1B)
                // multiplier(4B) + right_shift(1B)
                return do_bias ? 9 : 5;
            }


        protected:
            CV18xx(module::Chip chip) {}
            virtual ~CV18xx() {}
            static CV18xx *cv18xx;
            void load_ctx(module::Chip chip) {}
            cvk_context_t *cvk_ctx_;
            uint8_t tdmaBaseSelects[MAX_GLOBAL_MEMORY_REGION];
            std::vector<uint8_t> cmdbuf_;
            std::vector<uint8_t> cvk_cmd_buf_;
//            cvikernel_register dl_cvikernel_register;
        };
    }
}