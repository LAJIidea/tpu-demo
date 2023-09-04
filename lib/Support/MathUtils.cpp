//
// Created by 王奥博 on 2023/8/29.
//

#include "float.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "math_utils"

namespace tpu_mlir {



    template <typename T>
    int64_t to_int(T v, RoundingMode round_mode) {
        int64_t i64_val;
        if (round_mode == ROUNDING_HALF_AWAY_FROM_ZERO) {
            i64_val = std::round(v);
        } else if (round_mode == ROUNDING_DOWN) {
            i64_val = (int64_t)v;
        } else if (round_mode == ROUNDING_HALF_TO_EVEN) {
            float fraction, integer;
            float abs_v = std::abs(v);
            fraction = std::modf(abs_v, &integer);
            i64_val = (int64_t)integer;
            if (fraction > 0.5) {
                i64_val = i64_val + 1;
            } else if (fraction == 0.5) {
                if (i64_val & 0x01) {
                    i64_val = i64_val + 1;
                }
            }
            if (v < 0) {
                i64_val = -i64_val;
            }
        } else if (round_mode == ROUNDING_HALF_UP) {
            i64_val = std::floor(v + 0.5);
        } else if (round_mode == ROUNDING_HALF_DOWN) {
            i64_val = std::ceil(v - 0.5);
        } else {
            llvm_unreachable("not support round_mode.");
        }
        return i64_val;
    }

    template <typename T>
    int64_t saturate(T v, mlir::Type type, RoundingMode round_mode) {
        auto itype = llvm::dyn_cast<mlir::IntegerType>(type);
        if (!itype) {
            type.dump();
            llvm_unreachable("not support type");
        }
        int64_t max, min;
        auto N = itype.getWidth();
        if (itype.isUnsigned()) {
            max = llvm::maxUIntN(N);
            min = 0;
        } else {
            max = llvm::maxIntN(N);
            min = llvm::minIntN(N);
        }
        v = to_int(v, round_mode);
        if (v > max) {
            v = max;
        } else if (v < min) {
            v = min;
        }
        return v;
    }

    template <typename T>
    static int remove_value(std::vector<T> &v, int value) {
        int idx = 0;
        for (auto iter = v.begin(); iter != v.end(); iter++, idx++) {
            if (*iter == value) {
                v.erase(iter);
                return idx;
            }
        }
        return -1;
    }

    template int64_t to_int<float>(float v, RoundingMode round_mode);
    template int64_t to_int<long>(long v, RoundingMode round_mode);
    template int64_t to_int<double>(double v, RoundingMode round_mode);
    template int64_t to_int<int>(int v, RoundingMode round_mode);

    template int64_t saturate<float>(float v, mlir::Type type,
                                     RoundingMode round_mode);

    template int64_t saturate<int>(int v, mlir::Type type, RoundingMode round_mode);
    template int64_t saturate<long>(long v, mlir::Type type,
                                    RoundingMode round_mode);
    template int64_t saturate<double>(double v, mlir::Type type,
                                      RoundingMode round_mode);

    static void refresh(std::vector<int64_t> &order, int idx) {
        for (auto &v : order) {
            if (v > idx) {
                v--;
            }
        }
    }

    void function_relu(float *src, float *dst, int64_t size, float relu_limit, mlir::Type elem_type) {
#pragma omp parallel for schedule(static, omp_schedule(size))
        for (int64_t i = 0; i < size; ++i) {
            dst[i] = src[i] > 0 ? src[i] : 0;
            if (relu_limit > 0.f && dst[i] > relu_limit) {
                dst[i] = relu_limit;
            }
            if (elem_type && elem_type.isa<mlir::IntegerType>()) {
                dst[i] = saturate(dst[i], elem_type);
            }
        }
    }

    bool
    permute_reset(const std::vector<int64_t> &shape, const std::vector<int64_t> &order, std::vector<int64_t> &to_shape,
                  std::vector<int64_t> &to_order, int to_dim) {
        to_order.assign(order.begin(), order.end());
        to_shape.assign(shape.begin(), shape.end());
        int num_dims = shape.size();
        if (num_dims == to_dim) {
            return true;
        }
        if (num_dims > to_dim) {
            // remove dims = 1
            while (num_dims > to_dim) {
                int idx = remove_value(to_shape, 1);
                if (idx < 0) {
                    break;
                }
                remove_value(to_order, idx);
                refresh(to_order, idx);
                num_dims--;
            }
            // remove continous order
            while (num_dims > to_dim) {
                bool done = false;
                for (int i = 0; i < num_dims - 1; i++) {
                    if (to_order[i] + 1 == to_order[i + 1]) {
                        int idx = to_order[i];
                        to_shape[idx] *= to_shape[idx + 1];
                        to_shape.erase(to_shape.begin() + idx + 1);
                        to_order.erase(to_order.begin() + i + 1);
                        refresh(to_order, idx + 1);
                        num_dims--;
                        done = true;
                        break;
                    }
                }
                if (done == false) {
                    break;
                }
            }
            if (num_dims > to_dim) {
                return false;
            }
        } else if (num_dims < to_dim) {
            // reshape to  to_dims
            int inserted_dims = to_dim - num_dims;
            for (int i = 0; i < inserted_dims; i++) {
                to_shape.insert(to_shape.begin(), 1);
                to_order.insert(to_order.begin(), i);
            }

            for (int i = inserted_dims; i < to_dim; i++) {
                to_order[i] += inserted_dims;
            }
        }
        return true;
    }

    template<typename T>
    void function_permute(T *from, T *to, const std::vector<int64_t> &shape, const std::vector<int64_t> &order) {
        std::vector<int64_t> shape_6 = shape;
        std::vector<int64_t> order_6 = order;
        // convert to 6-dim
        for (int dim = shape.size(); dim < 6; ++dim) {
            shape_6.push_back(1);
            order_6.push_back(dim);
        }
        int64_t in = shape_6[0], ic = shape_6[1], it = shape_6[2], id = shape_6[3],
                ih = shape_6[4], iw = shape_6[5];
        int64_t o0 = order_6[0], o1 = order_6[1], o2 = order_6[2], o3 = order_6[3],
                o4 = order_6[4], o5 = order_6[5];
        for (int n = 0; n < in; ++n) {
            for (int c = 0; c < ic; ++c) {
                for (int t = 0; t < it; ++t) {
                    for (int d = 0; d < id; ++d) {
                        for (int h = 0; h < ih; h++) {
                            for (int w = 0; w < iw; w++) {
                                int cur[6] = {n, c, t, d, h, w};
                                int in_idx = w + h * iw + d * iw * ih + t * id * ih * iw +
                                             c * it * id * ih * iw + n * ic * it * id * ih * iw;
                                int out_idx = cur[o5] + cur[o4] * shape_6[o5] +
                                              cur[o3] * shape_6[o5] * shape_6[o4] +
                                              cur[o2] * shape_6[o5] * shape_6[o4] * shape_6[o3] +
                                              cur[o1] * shape_6[o5] * shape_6[o4] * shape_6[o3] *
                                              shape_6[o2] +
                                              cur[o0] * shape_6[o5] * shape_6[o4] * shape_6[o3] *
                                              shape_6[o2] * shape_6[o1];
                                to[out_idx] = from[in_idx];
                            }
                        }
                    }
                }
            }
        }
    }

    template void function_permute(float *from, float *to,
                                   const std::vector<int64_t> &shape,
                                   const std::vector<int64_t> &order);

    template void function_permute(uint16_t *from, uint16_t *to,
                                   const std::vector<int64_t> &shape,
                                   const std::vector<int64_t> &order);

    template void function_permute(uint8_t *from, uint8_t *to,
                                   const std::vector<int64_t> &shape,
                                   const std::vector<int64_t> &order);
}