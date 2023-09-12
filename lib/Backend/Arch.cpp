//
// Created by 王奥博 on 2023/8/29.
//

#include "tpu_mlir/Backend/Arch.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"

using namespace tpu_mlir::backend;

int64_t Arch::NPU_NUM = 0;
int64_t Arch::EU_BYTES = 0;
int64_t Arch::LMEM_BYTES = 0;
int64_t Arch::LMEM_BANKS = 0;
int64_t Arch::LMEM_BANK_BYTES = 0;
bool Arch::ALIGN_4N = false;
llvm::StringRef Arch::LIB_BACKEND_NAME = "";
module::Chip Arch::chip;
uint64_t Arch::FREQ = 0;
Arch *Arch::inst = nullptr;

void Arch::init(uint64_t freq) {
    if (inst != nullptr) {
        return;
    }

    chip = module::getChip();
    if (chip == module::Chip::ALL) {
        // do nothing
        return;
    } else {
        Arch::FREQ = freq;
        if (chip == module::Chip::BM1684) {
//            inst = &BM1684::instance();
        } else if (chip == module::Chip::BM1684X) {
//            inst = &BM1684X::instance();
        } else if (chip == module::Chip::BM1686) {
//            inst = &BM1686::instance(A2_1::value);
        } else if (chip == module::Chip::CV186X) {
//            inst = &BM1686::instance(A2_2::value);
        } else if (module::isCV18xx()) {
//            inst = &CV18xx::instance(chip);
        } else {
            llvm_unreachable("unsupport chip");
        }
    }
}

int64_t Arch::eu_num(double dbytes) { return EU_BYTES / dbytes; }

Arch::~Arch() {}

void Arch::load_library() {
    if (!DL.isValid()) {
        std::string Err;
        DL = llvm::sys::DynamicLibrary::getPermanentLibrary(LIB_BACKEND_NAME.data(),
                                                            &Err);
        if (DL.isValid() == false) {
            llvm_unreachable(Err.c_str());
        }
    }
}