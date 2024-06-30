import numpy as np
from . import cuda

class Tensor:
    def __init__(self, data, device='cpu', requires_grad=False, shape=None):
        # print('Creating tensor')
        assert device == 'cpu' or isinstance(data, Tensor) or (device == 'cuda' and shape is not None), 'shape must be provided for cuda tensors'
        self._device = device
        self._requires_grad = requires_grad
        self._grad = None

        if isinstance(data, Tensor):
            # If data is already a Tensor, copy its attributes
            self._data = data._data
            self._shape = data._shape
            self._device = data._device
        elif device == 'cpu':
            if isinstance(data, np.ndarray):
                self._data = data
            else:
                self._data = np.array(data)
            self._shape = self._data.shape
        elif device == 'cuda':
            self._data = data
            self._shape = shape
        else:
            raise ValueError(f"Unsupported device: {device}")
        
        # print(f'Created tensor on {self._device} with shape {self._shape}, {self._data = }')
    
    def __del__(self):
        if self._device == 'cuda':
            cuda.free_gpu_memory(self._data)
        
    @property
    def data(self):
        return self._data
    
    @property
    def device(self):
        return self._device
    
    @property
    def requires_grad(self):
        return self._requires_grad
    
    @property
    def grad(self):
        return self._grad

    @property
    def shape(self):
        return self._shape

    def to(self, device):
        # print(f"Moving tensor to {device}")
        if device == self._device:
            return self
        elif device == 'cpu':
            # print(f"type of self._data before moving: {type(self._data)} (if int, then it's a pointer)")
            data = cuda.gpu_to_cpu(self._data, self._shape)
            # print(f"{data.dtype = }")
            # print(f'data[:5]: {data[0, :5] if data.ndim > 1 else data[:5]}')
        elif device == 'cuda':
            # print(f"{self._data.dtype = }")
            # print(f'data[:5]: {self._data[0, :5] if self._data.ndim > 1 else self._data[:5]}')
            data = cuda.cpu_to_gpu(self._data)
        else:
            raise ValueError(f"Unsupported device: {device}")
        
        new_tensor = self._create_new(data, device)
        if self._grad is not None:
            new_tensor._grad = self._grad.to(device)
        return new_tensor

    def _create_new(self, data, device):
        return Tensor(data, shape=self.shape, device=device, requires_grad=self._requires_grad)
    
    def numel(self):
        if self._device == 'cpu':
            return self._data.size
        else:
            # TODO: count param size if data is on cuda
            return cuda.numel(self._data)