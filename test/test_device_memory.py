import unittest
import numpy as np
import quantumgrad.cuda as cuda
from quantumgrad import Tensor 
import quantumgrad.nn as nn
from quantumgrad.nn.parameter import Parameter

class TestDeviceMemory(unittest.TestCase):
    def test_cpu_to_device_to_cpu(self):
        a = np.random.rand(100, 100).astype(np.float32)
        device_a = cuda.cpu_to_gpu(a)
        cpu_a = cuda.gpu_to_cpu(device_a, (100, 100))
        np.testing.assert_allclose(a, cpu_a, rtol=1e-5, atol=1e-5)
    
    def test_tensor_to_device(self):
        a = Tensor(np.random.rand(100, 100).astype(np.float32), device='cpu')
        device_a = a.to('cuda')
        self.assertEqual(device_a.device, 'cuda')
        self.assertEqual(device_a.shape, a.shape)
        cpu_a = device_a.to('cpu')
        self.assertEqual(cpu_a.device, 'cpu')
        self.assertEqual(cpu_a.shape, a.shape)
        np.testing.assert_allclose(a.data, cpu_a.data, rtol=1e-5, atol=1e-5)

    def test_parameter_to_device(self):
        p = Parameter(np.random.rand(100, 100).astype(np.float32), device='cpu')
        device_p = p.to('cuda')
        self.assertEqual(device_p.device, 'cuda')
        self.assertEqual(device_p.shape, p.shape)
        cpu_p = device_p.to('cpu')
        self.assertEqual(cpu_p.device, 'cpu')
        self.assertEqual(cpu_p.shape, p.shape)
        np.testing.assert_allclose(p.data, cpu_p.data, rtol=1e-5, atol=1e-5)

    def test_linear_to_device(self):
        linear = nn.Linear(100, 100)
        # print('moving linear to cuda first time')
        device_linear = linear.to('cuda')
        print(f'in test, device_linear.weight.device: {device_linear.weight.device}')
        self.assertEqual(device_linear.weight.device, 'cuda')
        cpu_linear = device_linear.to('cpu')
        self.assertEqual(cpu_linear.weight.device, 'cpu')
        # np.testing.assert_allclose(linear.weight.data, cpu_linear.weight.data, rtol=1e-5, atol=1e-5)

    def test_large_tensor_to_device(self):
        a = Tensor(np.random.rand(1000, 1000).astype(np.float32), device='cpu')
        device_a = a.to('cuda')
        self.assertEqual(device_a.device, 'cuda')
        cpu_a = device_a.to('cpu')
        np.testing.assert_allclose(a.data, cpu_a.data, rtol=1e-5, atol=1e-5)

    def test_module_with_multiple_parameters_to_device(self):
        class SimpleModule(nn.Module):
            def __init__(self):
                super(SimpleModule, self).__init__()
                self.param1 = Parameter(np.random.rand(100, 100).astype(np.float32))
                self.param2 = Parameter(np.random.rand(50, 50).astype(np.float32))
                self.add_parameter(self.param1)
                self.add_parameter(self.param2)

            def forward(self, x):
                pass

        module = SimpleModule()
        device_module = module.to('cuda')
        self.assertEqual(device_module.param1.device, 'cuda')
        self.assertEqual(device_module.param2.device, 'cuda')
        cpu_module = device_module.to('cpu')
        self.assertEqual(cpu_module.param1.device, 'cpu')
        self.assertEqual(cpu_module.param2.device, 'cpu')
        np.testing.assert_allclose(module.param1.data, cpu_module.param1.data, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(module.param2.data, cpu_module.param2.data, rtol=1e-5, atol=1e-5)

if __name__ == '__main__':
    unittest.main()
