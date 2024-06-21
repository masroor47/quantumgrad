import unittest
import numpy as np
import quantumgrad.cuda as cuda

class TestDeviceMemory(unittest.TestCase):
    def test_cpu_to_device_to_cpu(self):
        a = np.random.rand(100, 100).astype(np.float32)
        device_a = cuda.cpu_to_gpu(a)
        cpu_a = cuda.gpu_to_cpu(device_a, (100, 100))

        np.testing.assert_allclose(a, cpu_a, rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
    unittest.main()
