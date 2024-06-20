import quantumgrad.nn as nn
from quantumgrad.nonlinearities.relu import ReLU

import numpy as np

class BobNet(nn.Module):
    def __init__(self, in_nodes, out_nodes):
        super(BobNet, self).__init__()
        self.fc1 = nn.Linear(in_nodes, 128)
        self.fc2 = nn.Linear(128, out_nodes)
    
    def forward(self, x):
        x = ReLU(self.fc1(x))
        return ReLU(self.fc2(x))

if __name__ == '__main__':
    device = 'cuda'

    bobNet = BobNet(784, 10)
    print(f"bobNet device: {bobNet._device}")
    print(f"bobNet data: {bobNet.fc1.weight._data}")
    print(f"{len(bobNet.parameters()) = } (weights and biases for two linear layers)\n")

    # can't count all params if device is cuda
    total_params = sum(p.numel() for p in bobNet.parameters())
    print(f"{total_params = }\n")

    bobNet = bobNet.to(device)

    num_weights = 784 * 128 + 128 * 10
    num_biases = 128 + 10
    num_floats_in_model = num_weights + num_biases
    print(f'{num_floats_in_model = }')
    megabytes_in_model = num_floats_in_model * 4 / 1024
    print(f"you are using {megabytes_in_model:.2f}MB on device {device}")

    import time
    time.sleep(3)

    # TODO: move back to cpu when initially on gpu
    # bobNet = bobNet.to('cpu')