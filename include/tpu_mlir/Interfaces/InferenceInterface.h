//
// Created by 王奥博 on 2023/8/28.
//
#pragma once

#include "mlir/IR/OpDefinition.h"

namespace tpu_mlir {
    struct InferenceParameter {
        std::vector<float *> inputs;
        std::vector<float *> outputs;
        void *handle = nullptr;
    };
} // namespace tpu_mlir

#include "tpu_mlir/Interfaces/InferenceInterface.h.inc"
