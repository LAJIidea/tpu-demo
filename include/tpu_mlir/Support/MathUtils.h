//
// Created by 王奥博 on 2023/8/29.
//

#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"

namespace tpu_mlir {
    // =======================
    // constant
    // =======================
    static constexpr double QMAX_INT8 = 127.0;
    static constexpr int BITS_INT8 = 8;

    // =======================
    // round mode
    // =======================
    typedef enum {
        ROUNDING_HALF_AWAY_FROM_ZERO = 0, // 1.5 -> 2   -1.5 -> -2
        ROUNDING_HALF_UP = 1,             // 1.5 -> 2   -1.5 -> -1
        ROUNDING_HALF_DOWN = 2,           // 1.5 -> 1   -1.5 -> -2
        ROUNDING_HALF_TO_EVEN = 3,        // 1.5 -> 2    2.5 -> 2
        ROUNDING_HALF_TO_ODD = 4,         // 1.5 -> 1    0.5 -> 1
        ROUNDING_HALF_TOWARDS_ZERO = 5,   // 1.5 -> 1   -1.5 -> -1
        ROUNDING_TOWARDS_ZERO = 6,        // 1.6 -> 1   -1.6 -> -1
        ROUNDING_AWAY_FROM_ZERO = 7,      // 1.4 -> 2   -1.4 -> -2
        ROUNDING_UP = 8,
        /* CEIL */ // 1.4 -> 2   -1.6 -> -1
        ROUNDING_DOWN = 9,
        /* FLOOR */ // 1.6 -> 1   -1.4 -> -2
        ROUNDING_UNKNOWN = -1
    } RoundingMode;

    // =======================
    // interfece for inference
    // =======================
    int omp_schedule(int count);

    template <typename T>
    int64_t to_int(T v, RoundingMode round_mode);
    template <typename T>
    int64_t saturate(T v, mlir::Type type,
                     RoundingMode round_mode = ROUNDING_HALF_AWAY_FROM_ZERO);
    extern template int64_t saturate<float>(float v, mlir::Type type,
                                            RoundingMode round_mode);
    extern template int64_t saturate<double>(double v, mlir::Type type,
                                             RoundingMode round_mode);

    void function_relu(float *src, float *dst, int64_t size, float relu_limit = 0.f,
                       mlir::Type elem_type = nullptr);

    template <typename T>
    T RightShiftRound(T src, int shift_num, RoundingMode round_mode);
    int32_t MultiplyByQuantizedMultiplier(
            int32_t x, int32_t multiplier, int shift,
            RoundingMode rmode = ROUNDING_HALF_AWAY_FROM_ZERO);
    int64_t applyMultiplierAndRShift(
            int64_t v, int64_t multiplier, int64_t rshift,
            tpu::RequantMode qmode = tpu::RequantMode::MultiplierShift,
            RoundingMode rmode = ROUNDING_HALF_UP);

    void tensor_sub_zp(float *tensor_after_zp, float *src, int64_t length,
                       float zero_point);
    void tensor_hw_transpose(float *dst, float *src, int64_t N, int64_t C,
                             int64_t H, int64_t W);
    void tensor_hc_transpose(float *dst, float *src, int64_t N, int64_t C,
                             int64_t H, int64_t W);
    void tensor_split(float *src_data, std::vector<std::vector<float>> &dst_data,
                      std::vector<int64_t> &shape, int slice_num, int axis);
    template <typename T>
    std::shared_ptr<std::vector<T>>
    tensor_slice(T *src_data, const std::vector<int64_t> &shape, int64_t axis,
                 int64_t offset, int64_t length);

    template <typename T>
    std::vector<int64_t> shape_expand_dim(llvm::ArrayRef<T> shape, int dims);
    template <typename T>
    std::vector<int64_t> shape_expand_dim(const std::vector<T> &shape, int dims);

    // reset permtue to 4dim or 5dim
    bool permute_reset(const std::vector<int64_t> &shape,
                       const std::vector<int64_t> &order,
                       std::vector<int64_t> &to_shape,
                       std::vector<int64_t> &to_order, int to_dim);

    template <typename T>
    void function_permute(T *from, T *to, const std::vector<int64_t> &shape,
                          const std::vector<int64_t> &order);
}
