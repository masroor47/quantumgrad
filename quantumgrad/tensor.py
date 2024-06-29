import numpy as np
from . import cuda

class Tensor:
    def __init__(self, data, device='cpu', requires_grad=False, shape=None):
        # print('Creating tensor')
        assert device == 'cpu' or (device == 'cuda' and shape is not None), 'shape must be provided for cuda tensors'
        self._device = device
        self._requires_grad = requires_grad
        self._grad = None
        # self._shape = data.shape if isinstance(data, np.ndarray) else None

        if device == 'cpu':
            if isinstance(data, np.ndarray):
                self._data = data
                self._shape = self._data.shape
            else:
                # print('I guess this is not an ndarray')
                self._data = np.array(data)
                self._shape = self._data.shape
        elif device == 'cuda':
            self._data = data
            # self._data = cuda.allocate_device_memory(data)
            self._shape = shape
        else:
            raise ValueError(f"Unsupported device: {device}")
        
        print(f'Created tensor on {device} with shape {self._shape}')
    
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
        print(f"Moving tensor to {device}")
        if device == self._device:
            return self
        elif device == 'cpu':
            print(f"type of self._data: {type(self._data)}")
            data = cuda.gpu_to_cpu(self._data, self._shape)
        elif device == 'cuda':
            data = cuda.cpu_to_gpu(self._data)
        else:
            raise ValueError(f"Unsupported device: {device}")
        
        # self._device = device

        new_tensor = Tensor(data, device=device, requires_grad=self._requires_grad, shape=self._shape)
        if self._grad is not None:
            new_tensor._grad = self._grad.to(device)
        return new_tensor
    
    def numel(self):
        if self._device == 'cpu':
            return self._data.size
        else:
            # TODO: count param size if data is on cuda
            return cuda.numel(self._data)