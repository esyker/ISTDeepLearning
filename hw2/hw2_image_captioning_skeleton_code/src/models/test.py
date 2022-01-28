import torch
import numpy as np

a = torch.tensor(np.array([[[1, 2, 3], [4, 5, 6]],[[1, 2, 3], [4, 5, 6]]]))
print(a.shape)
b = torch.tensor(np.array([[[1], [4]],[[3], [4]]]))
print(b.shape)
c = a*b
print(c.shape)
print(c)
d = c.sum(dim=1)
print(d.shape)
print(d)