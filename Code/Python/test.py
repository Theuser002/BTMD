import numpy as np
import torch
import torch.nn as nn

from torch.nn import CrossEntropyLoss

if __name__ == "__main__":
    print('TEST')
    # arr = np.random.randint(1, 3, size=(1, 3))
    # print(arr)
# t = torch.tensor([[[1, 2],[3, 4]],[[5, 6],[7, 8]]])
# rand = np.random.rand(10, 12)
# rand = torch.tensor(rand)
# print(rand.size())
# rand  = rand.detach().cpu().numpy()
# print(rand, rand.shape)

loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.randn(3, 5).softmax(dim=1)
print(input, input.shape, type(input))
print(target, target.shape, type(target))
output = loss(input, target)
output.backward()
