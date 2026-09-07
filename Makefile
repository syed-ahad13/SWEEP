# SWEEP — build file
#
# There is no default target. Name the binary you want:
#
#   make bin/<name> ARCH=sm_89     # from tests/|bench/|apps/|study/<name>.cu
#   make test_cpu                  # host-only; the only target that works on the Mac
#   make clean
#
# ARCH must be passed explicitly on every device that is not the daily 4090.
#
#   Device            ARCH        Note
#   ----------------  ----------  ---------------------------------
#   RTX 4090          sm_89       daily driver (default below)
#   Tesla T4          sm_75       Kaggle
#   Tesla P100        sm_60       Kaggle
#   A100 80GB         sm_80       HBM contrast
#   RTX 5090          sm_120      requires CUDA >= 12.8
#   RTX 3090          sm_86       within-Ampere control
#
ARCH ?= sm_89

# Pinned CUB/CCCL checkout. Empty = use the headers that ship with the toolkit.
# When set, these three include paths go BEFORE -Iinclude and before the
# toolkit's own include directory, so the pinned headers win over the
# toolkit's (needed in Step 22 for CUB's determinism API).
# Never bump this mid-study.
CCCL_DIR ?=
ifneq ($(strip $(CCCL_DIR)),)
CCCL_INC := -I$(CCCL_DIR)/cub -I$(CCCL_DIR)/thrust -I$(CCCL_DIR)/libcudacxx/include
else
CCCL_INC :=
endif

# NOTE: -use_fast_math must NEVER appear in this file, on any target.
# include/sweep/numerics.cuh and everything in study/ depend on exact IEEE-754
# rounding: two_sum / two_prod / df_add / df_mul recover the rounding error of
# each operation, and fast-math contraction (FMA fusion, reassociation,
# flush-to-zero, reciprocal approximation) silently deletes the correction
# terms. The code still compiles and still runs; the error bounds it is
# measuring just quietly become wrong.
# $(CCCL_INC) is placed ahead of -Iinclude so a pinned checkout wins over any
# other include path, the toolkit's implicit one included. (The plan writes this
# line with $(CCCL_INC) trailing; hoisting it is what makes the stated override
# rule true literally as well as in effect.)
NVCC = nvcc -O3 -std=c++17 -arch=$(ARCH) -lineinfo $(CCCL_INC) -Iinclude

CXXFLAGS_HOST = -std=c++17 -O2 -Iinclude

# Pattern rules, tried in this order: tests/ bench/ apps/ study/.
# GNU make picks the first pattern rule whose prerequisite actually exists,
# so binary names must stay unique across those four directories.
bin/%: tests/%.cu | bin
	$(NVCC) $< -o $@

bin/%: bench/%.cu | bin
	$(NVCC) $< -o $@

bin/%: apps/%.cu | bin
	$(NVCC) $< -o $@

bin/%: study/%.cu | bin
	$(NVCC) $< -o $@

bin:
	mkdir -p bin

# Host-only test of the CPU oracle / helper logic in tests/common.hpp.
# This is the one target that must build on a machine with no CUDA.
.PHONY: test_cpu
test_cpu: bin/test_common

bin/test_common: tests/test_common.cpp | bin
	$(CXX) $(CXXFLAGS_HOST) $< -o $@

.PHONY: clean
clean:
	rm -rf bin
