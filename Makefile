CUDA_SRC_DIR := quantumgrad/src/cuda
CUDA_OBJ_DIR := lib
NVCC := nvcc

CUDA_SRCS := $(wildcard $(CUDA_SRC_DIR)/*.cu)
CUDA_OBJS := $(patsubst $(CUDA_SRC_DIR)/%.cu,$(CUDA_OBJ_DIR)/%.so,$(CUDA_SRCS))

.PHONY: all clean

all: $(CUDA_OBJS)

$(CUDA_OBJ_DIR)/%.so: $(CUDA_SRC_DIR)/%.cu
	$(NVCC) -o $@ -shared -Xcompiler -fPIC $<

clean:
	rm -f $(CUDA_OBJ_DIR)/*.so