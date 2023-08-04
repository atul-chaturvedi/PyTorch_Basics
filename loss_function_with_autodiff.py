import torch

x = torch.randn([20,1], requires_grad=True)

y = 3*x - 2
# = w*x + b  --> w = 3, b =2
w = torch.tensor([1.], requires_grad=True) # we are using w =1
b = torch.tensor([1.], requires_grad=True) # and b =1

y_hat = w*x  + b


# we are calculating loss i,e actual value - our value -> y - y_hat
# then we are squaring the loss
loss = torch.sum((y_hat - y)**2)

# we are getting the loss between 200-300
print(loss)

# we differtiating the loss  with respect to  w and b
loss.backward()

# with respect to  w and b
print(w.grad, b.grad)

# we are getting w.grad  betwen -80 - 160 and b.grad betwen 80 - 160
# we are getting w.grad negative, this means we have to increse the value of w
# we are getting b.grad positive, this means we have to decrese the value of b

