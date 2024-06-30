import quantumgrad.cuda as cuda
from quantumgrad import Tensor

class Parameter(Tensor):
    def __init__(self, data, device='cpu', requires_grad=True):
        print('Creating Parameter on device', device, 'data:\n', data)
        super().__init__(data, device=device, requires_grad=requires_grad)
    
    def to(self, device):
        new_tensor = super().to(device)
        return Parameter(new_tensor, requires_grad=self.requires_grad)

    # def __repr__(self) -> str:
    #     return f"Parameter data:\n{self.data}\n"
