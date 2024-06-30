import unittest
import numpy as np
import quantumgrad.cuda as cuda
from quantumgrad import Tensor, nn
# import quantumgrad.nn as nn
from quantumgrad.nn.parameter import Parameter

class TestDeviceMemory(unittest.TestCase):
    '''
    Test the memory management of tensors, parameters, and modules when moving them between devices.
    '''
    def test_cpu_to_device_to_cpu(self):
        # print('\n\n\n------ test cpu to device to cpu ------')
        a = np.random.rand(100, 100).astype(np.float32)
        device_a = cuda.cpu_to_gpu(a)
        cpu_a = cuda.gpu_to_cpu(device_a, (100, 100))
        np.testing.assert_allclose(a, cpu_a, rtol=1e-5, atol=1e-5)
    
    def test_tensor_to_device(self):
        # print('\n\n\n------ test tensor to device ------')
        a = Tensor(np.random.rand(100, 100).astype(np.float32), device='cpu')
        device_a = a.to('cuda')
        self.assertEqual(device_a.device, 'cuda')
        self.assertEqual(device_a.shape, a.shape)
        # print('\n\ndevice info after moving to cuda:')
        # print(device_a._device)
        # print(device_a._shape)
        # print(hex(device_a._data))
        # print('\n\n')
        cpu_a = device_a.to('cpu')
        self.assertEqual(cpu_a.device, 'cpu')
        self.assertEqual(cpu_a.shape, a.shape)
        np.testing.assert_allclose(a.data, cpu_a.data, rtol=1e-5, atol=1e-5)

    def test_parameter_to_device(self):
        # print('\n\n\n------ test parameter to device ------')
        p = Parameter(np.random.rand(100, 100).astype(np.float32), device='cpu')
        device_p = p.to('cuda')
        self.assertEqual(device_p.device, 'cuda')
        self.assertEqual(device_p.shape, p.shape)
        # print('\n\ndevice info after moving to cuda:')
        # print(device_p._device)
        # print(device_p._shape)
        # print(hex(device_p._data))
        # print('\n\n')
        cpu_p = device_p.to('cpu')
        self.assertEqual(cpu_p.device, 'cpu')
        self.assertEqual(cpu_p.shape, p.shape)
        np.testing.assert_allclose(p.data, cpu_p.data, rtol=1e-5, atol=1e-5)

    def test_linear_to_device(self):
        # print('\n\n\n------ test linear to device ------')
        linear = nn.Linear(100, 100)
        # copy the weights to variable for comparison
        cpu_weights_data = linear.weight.data.copy()
        linear.to('cuda')
        # print(f'in test, moved to device; {linear.weight.device = }\n\n')
        self.assertEqual(linear.weight.device, 'cuda')
        # cpu_linear = device_linear.to('cpu')
        linear.to('cpu')
        self.assertEqual(linear.weight.device, 'cpu')
        np.testing.assert_allclose(linear.weight.data, cpu_weights_data, rtol=1e-5, atol=1e-5)

    def test_large_tensor_to_device(self):
        # print('\n\n\n------ test large tensor to device ------')
        a = Tensor(np.random.rand(1000, 1000).astype(np.float32), device='cpu')
        device_a = a.to('cuda')
        self.assertEqual(device_a.device, 'cuda')
        cpu_a = device_a.to('cpu')
        np.testing.assert_allclose(a.data, cpu_a.data, rtol=1e-5, atol=1e-5)

    def test_module_with_multiple_linear_submodules_to_device(self):
        # TODO: rewrite this test to use in-place .to() instead of returning new object
        # print('\n\n\n------ test module with multiple parameters to device ------')
        class SimpleModule(nn.Module):
            def __init__(self):
                super(SimpleModule, self).__init__()
                self.fc1 = nn.Linear(100, 100)
                self.fc2 = nn.Linear(100, 100)                

            def forward(self, x):
                pass

        module = SimpleModule()
        cpu_weights1_data = module.fc1.weight.data.copy()
        cpu_weights2_data = module.fc2.weight.data.copy()
        module.to('cuda')
        self.assertEqual(module.fc1.weight.device, 'cuda')
        self.assertEqual(module.fc1.weight.device, 'cuda')
        module.to('cpu')
        self.assertEqual(module.fc1.weight.device, 'cpu')
        self.assertEqual(module.fc2.weight.device, 'cpu')
        np.testing.assert_allclose(module.fc1.weight.data, cpu_weights1_data, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(module.fc2.weight.data, cpu_weights2_data, rtol=1e-5, atol=1e-5)

if __name__ == '__main__':
    unittest.main()
