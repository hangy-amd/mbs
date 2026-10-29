# Makefile for the mbs example example.
#
# Produces TWO shared libraries:
#   1. libmbs_kernel.so   -- the C++ kernel, built from sources under kernel/
#   2. mbs_example*.so -- the pybind11 interface module, which LINKS AGAINST
#                            libmbs_kernel.so
#
# Plus one .hsaco code object per kernel/asm/*.amdgcn, loaded at runtime.
#
# All build artifacts go into $(BUILD_DIR) (./build). At import time the
# interface module finds the kernel library via an rpath pointing at it.


PYTHON      ?= python3
HIPCC ?= hipcc
LLVM_BIN   ?= /opt/rocm/llvm/bin
GPU_ARCH   ?= gfx950

BUILD_DIR   := build
BUILD_ABS := $(abspath $(BUILD_DIR))
KERNEL_DIR    := kernel
INTERFACE_DIR := interface

ASM_DIR    := $(KERNEL_DIR)/asm
ASM_SRCS   := $(wildcard $(ASM_DIR)/*.amdgcn)
ASM_HSACOS := $(patsubst $(ASM_DIR)/%.amdgcn,$(BUILD_DIR)/asm/%.hsaco,$(ASM_SRCS))

# Query torch for its include/library paths and ABI.
TORCH_INCLUDES := $(shell $(PYTHON) -c "import torch.utils.cpp_extension as c; print(' '.join('-I'+p for p in c.include_paths()))")
TORCH_LIBDIR   := $(shell $(PYTHON) -c "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
PY_INCLUDES    := $(shell $(PYTHON)-config --includes)
TORCH_ABI      := $(shell $(PYTHON) -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")
TORCH_LDFLAGS := -L$(TORCH_LIBDIR) -Wl,-rpath,$(TORCH_LIBDIR) \
                 -ltorch -ltorch_cpu -ltorch_python -lc10

HIP_COMMON_FLAGS := -O3 -std=c++20 -fPIC \
                    --offload-arch=$(GPU_ARCH) \
                    -D__HIP_PLATFORM_AMD__=1 -D__Float4_e2m1fn_x2 \
                    -D_GLIBCXX_USE_CXX11_ABI=$(TORCH_ABI) \
                    -Wno-unused-result

$(BUILD_DIR)/asm/%.hsaco: $(ASM_DIR)/%.amdgcn | $(BUILD_DIR)/asm
	$(LLVM_BIN)/clang -x assembler -target amdgcn-amd-amdhsa -mcpu=$(GPU_ARCH) -c $< -o $(@:.hsaco=.o)
	$(LLVM_BIN)/ld.lld -shared $(@:.hsaco=.o) -o $@
	rm -f $(@:.hsaco=.o)

.DEFAULT_GOAL := all
.PHONY: all gemm asm clean

############### build example module ###############################

MODULE_NAME := example
MODULE_SO   := $(BUILD_DIR)/$(MODULE_NAME)$(shell $(PYTHON)-config --extension-suffix)

KERNEL_LIB    := $(BUILD_DIR)/libmbs_kernel.so
KERNEL_SRCS := $(wildcard $(KERNEL_DIR)/*.cpp)

# 1. Kernel shared library -- raw data-pointer arithmetic, torch-free.
$(KERNEL_LIB): $(KERNEL_SRCS) $(KERNEL_DIR)/example.h $(ASM_HSACOS) | $(BUILD_DIR)
	$(HIPCC) $(HIP_COMMON_FLAGS) -I$(KERNEL_DIR) -DMBS_ASM_DIR='"$(BUILD_ABS)/asm"' \
	    -shared $(KERNEL_SRCS) -o $@
	@echo "Built $(KERNEL_LIB)"

# 2. Interface module, linked against the kernel library above.
# $(KERNEL_LIB) is an order-only prerequisite (after the '|'): it must exist
# before we can link, but since the module links dynamically with an rpath, a
# rebuild of the kernel .so should NOT force the module to relink -- import
# picks up the new kernel library at runtime regardless.
$(MODULE_SO): $(INTERFACE_DIR)/example_pybind.cpp $(KERNEL_DIR)/example.h | $(KERNEL_LIB)
	$(HIPCC) $(HIP_COMMON_FLAGS) \
        -DTORCH_EXTENSION_NAME=$(MODULE_NAME) \
        -I$(KERNEL_DIR) $(TORCH_INCLUDES) $(PY_INCLUDES) -shared \
	    $(INTERFACE_DIR)/example_pybind.cpp -o $@ \
	    -L$(BUILD_ABS) -Wl,-rpath,$(BUILD_ABS) -lmbs_kernel $(TORCH_LDFLAGS)
	@echo "Built $(MODULE_SO)"

############### build gemm module ###############################

# HIP MXFP4 A4W4 GEMM (gfx950) module.
GEMM_MODULE_NAME := mbs_gemm_a4w4
GEMM_MODULE_SO   := $(BUILD_DIR)/$(GEMM_MODULE_NAME)$(shell $(PYTHON)-config --extension-suffix)
GEMM_KERNEL_LIB  := $(BUILD_DIR)/libmbs_gemm_kernel.so

# Torch-free HIP kernel library (device code, raw pointers).
$(GEMM_KERNEL_LIB): $(KERNEL_DIR)/gemm_a4w4.hip $(KERNEL_DIR)/gemm_a4w4.h | $(BUILD_DIR)
	$(HIPCC) $(HIP_COMMON_FLAGS) -I$(KERNEL_DIR) -shared $(KERNEL_DIR)/gemm_a4w4.hip -o $@
	@echo "Built $(GEMM_KERNEL_LIB)"

# Torch-facing pybind module, linked against the kernel library (rpath here).
$(GEMM_MODULE_SO): $(INTERFACE_DIR)/gemm_a4w4_pybind.cpp $(KERNEL_DIR)/gemm_a4w4.h | $(GEMM_KERNEL_LIB)
	$(HIPCC) $(HIP_COMMON_FLAGS) \
        -DTORCH_EXTENSION_NAME=$(GEMM_MODULE_NAME) \
        -I$(KERNEL_DIR) $(TORCH_INCLUDES) $(PY_INCLUDES) -shared \
	    $(INTERFACE_DIR)/gemm_a4w4_pybind.cpp -o $@ \
	    -L$(BUILD_ABS) -Wl,-rpath,$(BUILD_ABS) -lmbs_gemm_kernel $(TORCH_LDFLAGS)
	@echo "Built $(GEMM_MODULE_SO)"

################################################################

all: $(MODULE_SO) gemm

gemm: $(GEMM_MODULE_SO)

asm: $(ASM_HSACOS)

$(BUILD_DIR) $(BUILD_DIR)/asm:
	mkdir -p $@

clean:
	rm -rf $(BUILD_DIR)
