from __future__ import print_function
import torch

# Katherine Lasonde
# Tensors are similar to NumPy’s with the addition 
# being that Tensors can also be used on a GPU to accelerate computing.

# print a randomly initialized matrix
x = torch.rand(5, 3)
print(x)



# construct a randomly initialized matrix
x = torch.rand(5, 3)
print(x)

# construct a matrix filled with zeroes and dtype long
x = torch.zeros(5, 3, dtype=torch.long)
print(x)

# create a tensor directly from data
x = torch.tensor([5.5, 3])
print(x)

x = x.new_ones(5, 3, dtype=torch.double)      # new_* methods take in sizes
print(x)

x = torch.randn_like(x, dtype=torch.float)    # override dtype!
print(x) 

# size
print("size: ")
print(x.size())

# operations 
# addition
y = torch.rand(5, 3)
print(x + y)

# also addition 
print(torch.add(x, y))


# Any operation that mutates a tensor in-place is post-fixed with an _. 
# For example: x.copy_(y), x.t_(), will change x.
print(x[:, 1])


# y.add_(x)
print(y)


