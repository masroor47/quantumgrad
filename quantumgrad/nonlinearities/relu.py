import numpy as np
import quantumgrad.nn as nn
import quantumgrad.cuda as cuda

class ReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        if self._device == 'cuda':
            return cuda.relu(input)
        return np.maximum(input, 0)