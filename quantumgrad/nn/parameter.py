import quantumgrad.cuda as cuda
from quantumgrad import Tensor

class Parameter(Tensor):
    def __init__(self, data, shape=None, device='cpu', requires_grad=True):
        print('Creating Parameter on device', device, '; data:\n', data)
        super().__init__(data, shape=shape, device=device, requires_grad=requires_grad)
    
    # def to(self, device):
    #     print(f"Moving parameter to {device}")
    #     if device == self._device:
    #         return self
    #     elif device == 'cpu':
    #         print(f"type of self._data before moving: {type(self._data)} (if int, then it's a pointer)")
    #         data = cuda.gpu_to_cpu(self._data, self._shape)
    #         print(f"{data.dtype = }")
    #         print(f'data[:5]: {data[0, :5] if data.ndim > 1 else data[:5]}')
    #     elif device == 'cuda':
    #         print(f"{self._data.dtype = }")
    #         print(f'data[:5]: {self._data[0, :5] if self._data.ndim > 1 else self._data[:5]}')
    #         data = cuda.cpu_to_gpu(self._data)
    #     else:
    #         raise ValueError(f"Unsupported device: {device}")
        
    #     new_param = Parameter(data, shape=self._shape, device=device, requires_grad=self._requires_grad)
    #     if self._grad is not None:
    #         new_param._grad = self._grad.to(device)
    #     return new_param
    def _create_new(self, data, device):
        return Parameter(data, shape=self.shape, device=device, requires_grad=self._requires_grad)
    # def __repr__(self) -> str:
    #     return f"Parameter data:\n{self.data}\n"
