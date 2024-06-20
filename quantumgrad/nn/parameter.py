import cuda 

class Parameter:
    def __init__(self, data):
        self._data = data
        self._device = 'cpu'

    def to(self, device):
        if device == 'cuda':
            self._data = cuda.cpu_to_gpu(self._data)
        else:
            self._data = cuda.gpu_to_cpu(self._data)
        self._device = device
        return self
    
    def numel(self):
        if self._device == 'cpu':
            return self._data.size
        else:
            # TODO: count param size if data is on cuda
            print("device is cuda, can't give you the size rn")
            raise NotImplementedError