import numpy as np
from . import cuda

class Tensor:
    def __init__(self, data, device='cpu'):
        self._device = device
        if device == 'cpu':
            self._data = np.array(data)
        elif device == 'cuda':
            self._data = cuda.allocate_device_memory(data)
        else:
            raise ValueError(f"Unsupported device: {device}")

    def to(self, device):
        if device == self._device:
            return self
        elif device == 'cpu':
            data = self._data.copy()
            return Tensor(data, device='cpu')
        elif device == 'cuda':
            data = cuda.tensor_to_device(self._data)
            return Tensor(data, device='cuda')
        else:
            raise ValueError(f"Unsupported device: {device}")