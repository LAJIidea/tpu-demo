//
// Created by 王奥博 on 2023/8/29.
//

#include "float.h"
#include "omp.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "llvm/Support/Debug.h"
#include <numeric>

#define DEBUG_TYPE "math_utils"

namespace tpu_mlir {

    int omp_schedule(int count) {
        return (count + omp_get_num_threads() - 1) / omp_get_num_threads();
    }

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

    template int64_t saturate<float>(float v, mlir::Type type,
                                     RoundingMode round_mode);

    template int64_t saturate<int>(int v, mlir::Type type, RoundingMode round_mode);
    template int64_t saturate<long>(long v, mlir::Type type,
                                    RoundingMode round_mode);
    template int64_t saturate<double>(double v, mlir::Type type,
                                      RoundingMode round_mode);

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

    template<typename T>
    T RightShiftRound(T src, int shift_num, RoundingMode round_mode) {
        if (shift_num == 0)
            return src;
        if (shift_num > 63)
            shift_num = 63;
        T val, res;
        if (shift_num < 0) {
            return src << (-shift_num);
        }
        val = src >> shift_num;
        res = val;
        T lo_mask = (1ull << shift_num) - 1;
        T mant = src & lo_mask;
        T mant_0d5 = 1ull << (shift_num - 1);
        if (round_mode == ROUNDING_HALF_TO_EVEN) {
            if (mant == mant_0d5)
                res = val + (val & 1);
            else if (mant > mant_0d5)
                res = val + 1;
        } else if (round_mode == ROUNDING_HALF_AWAY_FROM_ZERO) {
            if (src >= 0 && mant >= mant_0d5)
                res = val + 1;
            else if (src < 0 && mant > mant_0d5)
                res = val + 1;
        } else if (round_mode == ROUNDING_TOWARDS_ZERO) {
            if (src < 0)
                res = val + (mant != 0);
        } else if (round_mode == ROUNDING_DOWN)
            res = val;
        else if (round_mode == ROUNDING_UP)
            res = val + (mant != 0);
        else if (round_mode == ROUNDING_HALF_UP) {
            if (mant >= mant_0d5)
                res = val + 1;
        } else if (round_mode == ROUNDING_HALF_DOWN) {
            if (mant > mant_0d5)
                res = val + 1;
        }
        return res;
    }
    template long long RightShiftRound(long long src, int shift_num,
                                       RoundingMode round_mode);
    template int64_t RightShiftRound(int64_t src, int shift_num,
                                     RoundingMode round_mode);

    // to compilable with tflite
    // tensorflow/lite/kernels/internal/common.h:MultiplyByQuantizedMultiplier()
    int32_t MultiplyByQuantizedMultiplier(int32_t x, int32_t multiplier, int shift, RoundingMode rmode) {
        // int shift = -(rshift - 31);
        int64_t value = shift > 0 ? x << shift : x;
        value = RightShiftRound(value * multiplier, 31, ROUNDING_HALF_UP);
        if (value > (1ll << 31) - 1)
            value = (1ll << 31) - 1;
        else if (value < -(1ll << 31))
            value = -(1ll << 31);
        if (shift < 0) {
            value = RightShiftRound(value, -shift, rmode);
        }
        return (int32_t)value;
    }

    int64_t applyMultiplierAndRShift(int64_t v, int64_t multiplier, int64_t rshift, tpu::RequantMode qmode,
                                     RoundingMode rmode) {
        switch (qmode) {
            case tpu::RequantMode::MultiplierShift:
                if (module::isCV18xx()) {
                    return to_int(((((float)v * multiplier)) / (1 << rshift)), rmode);
                } else {
                    return RightShiftRound(v * multiplier, (int)rshift, rmode);
                }
            case tpu::RequantMode::OnlyShift:
                return RightShiftRound(v, (int)rshift, rmode);
            case tpu::RequantMode::QDM:
            case tpu::RequantMode::TFLite:
            case tpu::RequantMode::TFLite_LShift:
                if (module::isCV18xx()) {
                    rshift = -rshift;
                }
                return MultiplyByQuantizedMultiplier((int32_t)v, (int32_t)multiplier,
                                                     (int32_t)rshift, rmode);
        }
        llvm_unreachable("unsupport quant multiplier mode.");
        return 0;
    }

    void tensor_sub_zp(float *tensor_after_zp, float *src, int64_t length, float zero_point) {
#pragma omp parallel for schedule(static, omp_schedule(length))
        for (int i = 0; i < length; ++i) {
            tensor_after_zp[i] = src[i] - zero_point;
        }
    }

    void tensor_hw_transpose(float *dst, float *src, int64_t N, int64_t C, int64_t H, int64_t W) {
#pragma omp parallel for schedule(static, omp_schedule(N *C))
        for (int64_t nc = 0; nc < N * C; ++nc) {
            int64_t nc_offset = nc * H * W;
            for (int w = 0; w < W; ++w) {
                for (int h = 0; h < H; ++h) {
                    int64_t d_offset = nc_offset + w * H + h;
                    int64_t s_offset = nc_offset + h * W + w;
                    dst[d_offset] = src[s_offset];
                }
            }
        }
    }

    void tensor_hc_transpose(float *dst, float *src, int64_t N, int64_t C, int64_t H, int64_t W) {
#pragma omp parallel for schedule(static, omp_schedule(N))
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t h = 0; h < H; ++h) {
                for (int64_t c = 0; c < C; ++c) {
                    for (int64_t w = 0; w < W; ++w) {
                        int64_t s_offset = w + h * W + c * H * W + n * C * H * W;
                        int64_t d_offset = w + c * W + h * C * W + n * C * H * W;
                        dst[d_offset] = src[s_offset];
                    }
                }
            }
        }
    }

    void
    tensor_split(float *src_data, std::vector<std::vector<float>> &dst_data, std::vector<int64_t> &shape, int slice_num,
                 int axis) {
        assert(shape[axis] % slice_num == 0);
        assert(axis < shape.size());
        dst_data.resize(slice_num);

        // The data can be treated as 3 dim
        // 1.pre of the axis
        // 2.the axis
        // 3.behind of axis
        std::vector<int64_t> fake_shape(3);
        fake_shape[0] = std::accumulate(shape.begin(), shape.begin() + axis, 1,
                                        std::multiplies<int64_t>());
        fake_shape[1] = shape[axis];
        fake_shape[2] = std::accumulate(shape.begin() + axis + 1, shape.end(), 1,
                                        std::multiplies<int64_t>());
        std::vector<int64_t> fake_offset(3);
        fake_offset[2] = 1;
        fake_offset[1] = fake_offset[2] * fake_shape[2];
        fake_offset[0] = fake_offset[1] * fake_shape[1];

        int64_t indices = shape[1] / slice_num;
        int64_t slice_size = fake_shape[0] * indices * fake_offset[1];

        // each slice
#pragma omp parallel for schedule(static, omp_schedule(slice_num))
        for (int64_t i = 0; i < slice_num; ++i) {
            dst_data[i].resize(slice_size);
            // each fake dim 0
#pragma omp parallel for schedule(static, omp_schedule(fake_shape[0]))
            for (int64_t j = 0; j < fake_shape[0]; ++j) {
                float *src_ptr =
                        src_data + j * fake_offset[0] + i * indices * fake_offset[1];
                float *dst_ptr = dst_data[i].data() + j * indices * fake_offset[1];
                std::copy(src_ptr, src_ptr + indices * fake_offset[1], dst_ptr);
            }
        }
    }

    template<typename T>
    std::shared_ptr<std::vector<T>>
    tensor_slice(T *src_data, const std::vector<int64_t> &shape, int64_t axis, int64_t offset, int64_t length) {
        auto outer_size = std::accumulate(shape.begin(), shape.begin() + axis, 1,
                                          std::multiplies<int64_t>());
        auto axis_size = shape[axis];
        auto inner_size = std::accumulate(shape.begin() + axis + 1, shape.end(), 1,
                                          std::multiplies<int64_t>());
        assert(length + offset <= axis_size);
        auto output =
                std::make_shared<std::vector<T>>(outer_size * inner_size * length);
        for (int64_t i = 0; i < outer_size; i++) {
            T *src_ptr = src_data + i * axis_size * inner_size + offset * inner_size;
            T *dst_ptr = output->data() + i * length * inner_size;
            std::copy(src_ptr, src_ptr + length * inner_size, dst_ptr);
        }
        return output;
    }

    template std::shared_ptr<std::vector<float>>
    tensor_slice(float *src_data, const std::vector<int64_t> &shape, int64_t axis,
                 int64_t offset, int64_t length);

    template std::shared_ptr<std::vector<uint16_t>>
    tensor_slice(uint16_t *src_data, const std::vector<int64_t> &shape,
                 int64_t axis, int64_t offset, int64_t length);

    template std::shared_ptr<std::vector<int8_t>>
    tensor_slice(int8_t *src_data, const std::vector<int64_t> &shape, int64_t axis,
                 int64_t offset, int64_t length);

    template<typename T>
    std::vector<int64_t> tpu_mlir::shape_expand_dim(const std::vector<T> &shape, int dims) {
        int diff = dims - shape.size();
        std::vector<int64_t> shape_v(shape.begin(), shape.end());
        if (diff == 0)
            return shape_v;
        shape_v.insert(shape_v.begin(), diff, 1);
        return shape_v;
    }
    template std::vector<int64_t> shape_expand_dim(const std::vector<float> &shape,
                                                   int dims);

    template<typename T>
    std::vector<int64_t> shape_expand_dim(llvm::ArrayRef<T> shape, int dims) {
        int diff = dims - shape.size();
        std::vector<int64_t> shape_v(shape.begin(), shape.end());
        if (diff == 0)
            return shape_v;
        shape_v.insert(shape_v.begin(), diff, 1);
        return shape_v;
    }
    template std::vector<int64_t> shape_expand_dim(llvm::ArrayRef<float> shape,
                                                   int dims);
    template std::vector<int64_t> shape_expand_dim(llvm::ArrayRef<int64_t> shape,
                                                   int dims);

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