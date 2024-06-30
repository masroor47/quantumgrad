import quantumgrad.cuda as cuda
from quantumgrad import Tensor

class Parameter(Tensor):
    def __init__(self, data, shape=None, device='cpu', requires_grad=True):
        # print('Creating Parameter on device', device, '; data:\n', data)
        super().__init__(data, shape=shape, device=device, requires_grad=requires_grad)
    
    def _create_new(self, data, device):
        return Parameter(data, shape=self.shape, device=device, requires_grad=self._requires_grad)

    # def __repr__(self) -> str:
    #     return f"Parameter data:\n{self.data}\n"
