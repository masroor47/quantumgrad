import numpy as np
from . import cuda

class Tensor:
    def __init__(self, data, device='cpu', requires_grad=False):
        self._device = device
        self._requires_grad = requires_grad
        self._grad = None

        if device == 'cpu':
            self._data = np.array(data)
        elif device == 'cuda':
            self._data = cuda.allocate_device_memory(data)
        else:
            raise ValueError(f"Unsupported device: {device}")
        
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

    def to(self, device):
        if device == self._device:
            return self
        elif device == 'cpu':
            data = cuda.gpu_to_cpu(self._data)
        elif device == 'cuda':
            data = cuda.cpu_to_gpu(self._data)
        else:
            raise ValueError(f"Unsupported device: {device}")

        new_tensor = Tensor(data, device=device, requires_grad=self._requires_grad)
        if self._grad is not None:
            new_tensor._grad = self._grad.to(device)
        return new_tensor
    
    def numel(self):
        if self._device == 'cpu':
            return self._data.size
        else:
            # TODO: count param size if data is on cuda
            return cuda.numel(self._data)