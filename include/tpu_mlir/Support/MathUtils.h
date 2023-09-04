//
// Created by 王奥博 on 2023/8/29.
//

#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "tpu_mlir/Dialect/Top/IR/TopOps.h"

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

    template <typename T>
    int64_t to_int(T v, RoundingMode round_mode);
    template <typename T>
    int64_t saturate(T v, mlir::Type type,
                     RoundingMode round_mode = ROUNDING_HALF_AWAY_FROM_ZERO);

    void function_relu(float *src, float *dst, int64_t size, float relu_limit = 0.f,
                       mlir::Type elem_type = nullptr);

    // reset permtue to 4dim or 5dim
    bool permute_reset(const std::vector<int64_t> &shape,
                       const std::vector<int64_t> &order,
                       std::vector<int64_t> &to_shape,
                       std::vector<int64_t> &to_order, int to_dim);

    template <typename T>
    void function_permute(T *from, T *to, const std::vector<int64_t> &shape,
                          const std::vector<int64_t> &order);
}
