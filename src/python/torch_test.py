import torch


a = torch.ones((1, 5)).cuda()
b = torch.ones((1, 5)).cuda()
c = a + b
print(c)