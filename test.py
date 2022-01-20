import torch
from torch import tensor
from bdq import Network
from bdq import ReplayBuffer
from bdq import BDQ

"""
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
"""
"""
buffer = ReplayBuffer(2, 2, max_len=10, beta_increase_steps=10)
buffer.store_experience([0.1, 0.2], [0, 1], 0.0, [0.2, 0.3], 0)
buffer.store_experience([0.2, 0.2], [5, 4], 0.0, [0.2, 0.3], 0)

buffer.update_experiences([0], [2.0])
indices, weights, states, actions, rewards, next_states, terminals = buffer.get_experiences(1)

length = len(buffer)

pass
"""

agent = BDQ(2, (2, 2), (2, 2), (2, 2), batch_size=3, buffer_size_min=1, device='cuda')
agent.reset()
agent.experience([0.1, 0.2], [0, 1], 0.0, [0.2, 0.3], 0)
actions1 = agent.act([0.1, 0.2])
actions2 = agent.act_optimally([0.1, 0.2])
agent.train()