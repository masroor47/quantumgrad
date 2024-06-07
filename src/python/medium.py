# Python function calling the compiled C++/CUDA function

import ctypes

# Load the CUDA library
cuda_lib = ctypes.CDLL('./lib/sumArrayGPU.so')  # Update with the correct path

# Define the function prototype
cuda_lib.my_cuda_function.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_int]
cuda_lib.my_cuda_function.restype = None

# Prepare data
input_data = [1, 2, 3, 4]
output_data = [0, 0, 0, 0]
size = len(input_data)

# Convert Python lists to ctypes arrays
input_array = (ctypes.c_int * size)(*input_data)
output_array = (ctypes.c_int * size)(*output_data)

# Call the CUDA function
cuda_lib.my_cuda_function(input_array, output_array, size)

# Print the result
result = list(output_array)
print("Result:", result)