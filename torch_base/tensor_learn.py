import torch

x = torch.tensor([1,2,3,4])
y = torch.ones((3,1))
z = torch.rand((2,3))
print(x)
print(y)
print(z)
print(x+y)
x = torch.tensor(2.0, requires_grad=True)
y = x**2 + 3*x + 1
y.backward()
print(x.grad)  # dy/dx = 2x+3