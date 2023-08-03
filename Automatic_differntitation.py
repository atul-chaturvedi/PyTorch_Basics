import torch
x = torch.ones([3,3], requires_grad= True)
print(x)

y = x + 5
print(y)


z = y*y +1
print(z)

t = torch.sum(z)
print(t)

#below code menas 
# dt/dx = dz/dx = dz/dy * dy/dx = d/dx(y*y +1) * d/dx(x+5)
# => dt/dx  = 2y * 1
# when x = 1, y = (x+5) = 6
# then dt/dx = 2y = 2*6 = 12
t.backward()
print(x.grad)