import ctypes
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
cuda_lib_path = os.path.join(current_dir, '../../lib/matrix_ops.so')
cuda_lib = ctypes.cdll.LoadLibrary(cuda_lib_path)

cpu_to_gpu = cuda_lib.cpu_to_gpu
cpu_to_gpu.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
cpu_to_gpu.restype = ctypes.c_void_p

gpu_to_cpu = cuda_lib.gpu_to_cpu
gpu_to_cpu.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
gpu_to_cpu.restype = ctypes.POINTER(ctypes.c_float)