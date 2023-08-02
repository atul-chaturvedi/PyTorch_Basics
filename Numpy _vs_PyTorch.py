import time
import numpy as np
import torch

start_numpy = time.time()

for i in range(10):
    a = np.random.randn(10000, 10000)
    b = np.random.randn(10000, 10000)
    np.add(b,a)
    
end_numpy = time.time()

print("Numpy Time = ", end_numpy - start_numpy)


start_torch = time.time()

for i in range(100):
    a = torch.randn([1000, 1000])
    b = torch.randn([1000, 1000])
    b.add_(a)
    
end_torch = time.time()

print("Torch Time = ",end_torch - start_torch)


# with GPU support

# print(torch.cuda.device_count())
# print(torch.cuda.device(0))
# print(torch.cuda.get_device_name(0))

cuda0 =torch.device('cuda:0')


start_torch_cuda = time.time()

for i in range(100):
    a = torch.randn([1000, 1000], device = cuda0)
    b = torch.randn([1000, 1000], device = cuda0)
    b.add_(a)
    
end_torch_cuda = time.time()

print("GPU Torch Time = ",end_torch_cuda - start_torch_cuda)

