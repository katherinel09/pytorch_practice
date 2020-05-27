# KL
# blitz tutorial 

import torch

#  autograd package provides automatic differentiation for all operations on Tensors
# set its attribute .requires_grad as True => tracks all operations on it. 
# .backward() => all the gradients computed automatically
# The gradient for this tensor will be accumulated into .grad attribute.


# .detach() => stop a tensor from tracking history


# Tensor and Function are interconnected and build up an acyclic graph, 
# that encodes a complete history of computation. 


# each tensor has .grad_fn attribute that references a Function that has created the Tensor

# compute derivtives => .backward() on a tensor

# create a tensor and set require
x = torch.ones(2, 2, requires_grad=True)
print(x)


# operation
y = x + 2
print(y)


# printing the grad function 
print(y.grad_fn)

# more operations on y 
z = y * y * 3
out = z.mean()

print(z, out)


# creating a new random torch
print("creating tensor a: ")
a = torch.randn(2, 2)
print(a)

print("changing a: ")
a = ((a * 3) / (a - 1))
print(a)

print("a.requires_grad: " + str(a.requires_grad))

print("making grad true")
a.requires_grad_(True)


print("printing a.requires grad again " + str(a.requires_grad))

print("creating b from a")
b = (a * a).sum()

print("print the grad fcn of b")
print(b.grad_fn)

