import torch
import numpy as np

#initialize Tensors

x = torch.ones(3,2)
print(x)

x = torch.zeros(3,2)
print(x)

x = torch.rand(3,2)
print(x)


x = torch.empty(3,2)
print(x)

y = torch.zeros(3,2)
print(y)

x =torch.linspace(0, 1, steps = 10)
print(x)


x = torch.tensor([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]])

print(x)

#slicing Tensors

print(x.size())
print(x[:,1])
print(x[0,:])


y = x[1,1]
print(y)
print(y.item())


#reshaping tensors 
x = torch.tensor([[1, 2, 3],
                 [4, 5, 6]])
print(x)
y = x.view(3, 2)
print(y)


y = x.view(6,-1)
print(y)


# Simple Tensor Operations
x = torch.ones([3,2])
y = torch.ones([3,2])

z = x + y
print(z)

z = x - y
print(z)

z = x * y
print(z)

z = y.add(x)
print(z)
print(y)

z = y.add_(x) # midified inplace
print(z)
print(y)


