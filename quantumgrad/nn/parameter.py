import quantumgrad.cuda as cuda
from quantumgrad import Tensor

class Parameter(Tensor):
    def __init__(self, data, device='cpu'):
        super().__init__(data, device=device, requires_grad=True)

    def __repr__(self) -> str:
        return f"Parameter data:\n{self.data}\n"
