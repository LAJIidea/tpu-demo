//
// Created by 王奥博 on 2023/8/29.
//

#pragma once

#include "mlir/IR/Builders.h"
#include "tpu_mlir/Support/Module.h"
#include "llvm/Support/DynamicLibrary.h"
#include <assert.h>
#include <cmath>
#include <cstring>
#include <vector>

namespace tpu_mlir {
    namespace backend {
        using A2_1 = std::integral_constant<int, 900>;
        using A2_2 = std::integral_constant<int, 375>;

        class Arch {
        public:
            static void init(uint64_t freq);
            static int64_t NPU_NUM;
            static int64_t EU_BYTES;
            static int64_t LMEM_BYTES;
            static int64_t LMEM_BANKS;
            static int64_t LMEM_BANK_BYTES;
            static llvm::StringRef LIB_BACKEND_NAME;
            static bool ALIGN_4N;
            static module::Chip chip;
            static uint64_t FREQ;
            static uint64_t get_frequance() {return Arch::FREQ;}
            // dbytes is 0.5 for INT4
            static int64_t eu_num(double dbytes);
            static int64_t get_n_align(int64_t dtype_bytes) {
                return ALIGN_4N ? (4 / dtype_bytes) : 1;
            }

        protected:
            static Arch *inst;
            llvm::sys::DynamicLibrary DL;
            Arch() {};
            virtual ~Arch() = 0;
            void load_library();
        };
    }
}
