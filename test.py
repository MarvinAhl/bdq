import torch
from torch import tensor
from bdq import Network

net = Network(2, (2, 2), shared=(2, 2), branch=[2])
data = []

for param in net.parameters():
    data = param.data
    pass

states = tensor([[2.0, 1.0], [2.0, 1.0]])
qs = net(states)
loss = 0.0
for q in qs:
    loss += q.pow(2).mul(0.5).mean()
loss /= len(qs)
loss.backward()

print()
for param in net.parameters():
    pass

print()

for i, param in enumerate(net.parameters()):
    if i < 4:  # len(shared) * 2
        param.grad /= 3  # len(actions) + 1
    print(param.grad)

#print(net)
pass