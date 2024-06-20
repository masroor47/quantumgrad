CUDA_KERNELS_DIR := quantumgrad/cuda/kernels
CUDA_UTILS_DIR := quantumgrad/cuda/utils
CUDA_OBJ_DIR := lib
NVCC := nvcc

CUDA_KERNEL_SRCS := $(wildcard $(CUDA_KERNELS_DIR)/*.cu)
CUDA_UTILS_SRCS := $(wildcard $(CUDA_UTILS_DIR)/*.cu)

CUDA_KERNEL_OBJS := $(patsubst $(CUDA_KERNELS_DIR)/%.cu,$(CUDA_OBJ_DIR)/libcudakernels.so,$(CUDA_KERNEL_SRCS))
CUDA_UTILS_OBJS := $(patsubst $(CUDA_UTILS_DIR)/%.cu,$(CUDA_OBJ_DIR)/libcudautils.so,$(CUDA_UTILS_SRCS))

.PHONY: all clean

all: $(CUDA_KERNEL_OBJS) $(CUDA_UTILS_OBJS)

$(CUDA_OBJ_DIR)/libcudakernels.so: $(CUDA_KERNEL_SRCS)
	$(NVCC) -o $@ -shared -Xcompiler -fPIC $^

$(CUDA_OBJ_DIR)/libcudautils.so: $(CUDA_UTILS_SRCS)
	$(NVCC) -o $@ -shared -Xcompiler -fPIC $^

clean:
	rm -f $(CUDA_OBJ_DIR)/*.so