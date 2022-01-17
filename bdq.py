"""
Author: Marvin Ahlborn

This is an implementation of the Branching Dueling Q-Network (BDQ) Algorithm proposed
by Tavakoli et al. from the Imperial College London in their paper
'Action Branching Architectures for Deep Reinforcement Learning' in 2017.

BDQ expands Dueling DDQN with PER to use multiple action sets distributed across
individual Branches using a custom Network Architecture. This makes it scalable
to high dimensional action-spaces. For more information I highly recommend to
read their paper under: https://arxiv.org/abs/1711.08946
"""

import torch
from torch import tensor
from torch import nn
import numpy as np

class Network(nn.Module):
    """
    Branching Dueling Q-Network
    """
    def __init__(self, state, actions, shared=(512, 512), branch=(128, 128)):
        """
        state: Integer telling the state dimension
        actions: Tuple of integers where every int stands for the number of discretized subactions for every action
        shared: Tuple containing the neurons for the shared hidden layers
        branch: Tuple containing the neurons for the hidden layers per branch (advantage and state-value branches)
        """
        super(Network, self).__init__()

        self.actions = actions

        # Shared part of the Network
        shared_modules = nn.ModuleList()

        shared_modules.append(nn.Linear(state, shared[0]))
        shared_modules.append(nn.LeakyReLU(0.1))
        
        for i in range(len(shared) - 1):
            shared_modules.append(nn.Linear(shared[i], shared[i+1]))
            shared_modules.append(nn.LeakyReLU(0.1))
        
        self.shared_stack = nn.Sequential(*shared_modules)

        # State-Value branch
        value_modules = nn.ModuleList()

        value_modules.append(nn.Linear(shared[-1], branch[0]))
        value_modules.append(nn.LeakyReLU(0.1))

        for i in range(len(branch) - 1):
            value_modules.append(nn.Linear(branch[i], branch[i+1]))
            value_modules.append(nn.LeakyReLU(0.1))
        
        value_modules.append(nn.Linear(branch[-1], 1))  # Connection to value output

        self.value_stack = nn.Sequential(*value_modules)

        # Advantage branches
        self.branch_stacks = []

        for i in range(len(actions)):
            branch_modules = nn.ModuleList()

            branch_modules.append(nn.Linear(shared[-1], branch[0]))
            branch_modules.append(nn.LeakyReLU(0.1))

            for j in range(len(branch) - 1):
                branch_modules.append(nn.Linear(branch[j], branch[j+1]))
                branch_modules.append(nn.LeakyReLU(0.1))
            
            branch_modules.append(nn.Linear(branch[-1], actions[i]))  # Connection to current action

            self.branch_stacks.append(nn.Sequential(*branch_modules))
    
    def forward(self, state):
        """
        Forward pass of Branched Network.
        state: Tensor of a batch of states (Generally 2D)
        action_outputs: List of Tensors of a batch of actions (Every Tensor contains a batch of one particular action-value output)
        """
        shared_output = self.shared_stack(state)

        value_output = self.value_stack(shared_output)

        action_outputs = []
        for i in range(len(self.actions)):
            branch_output = self.branch_stacks[i](shared_output)  # The action-advantages

            # Q-Value Aggregation
            value_output_exp = value_output.expand_as(branch_output)
            action_output = value_output_exp + branch_output - branch_output.mean(1, keepdim=True).expand_as(branch_output)
            action_outputs.append(action_output)  # The Q-Values
        
        return action_outputs

class ReplayBuffer:
    """
    Prioritizing Replay Buffer
    """
    def __init__(self, max_len, state_dim, alpha=0.6, beta=0.1, beta_increase_steps=20000):
        self.states = np.empty((max_len, state_dim), dtype=np.float32)
        self.actions = np.empty(max_len, dtype=np.int16)
        self.rewards = np.empty(max_len, dtype=np.float32)
        self.next_states = np.empty((max_len, state_dim), dtype=np.float32)
        self.terminals = np.empty(max_len, dtype=np.int8)
        self.errors = np.empty(max_len, dtype=np.float32)  # The Q-Network temporal difference error used for training
        self.weights = np.empty(max_len, dtype=np.float32)  # Weights for gradient descend bias correction
        self.probabilities = np.empty(max_len, dtype=np.float32)

        self.index = 0
        self.full = False
        self.max_len = max_len
        self.rng = np.random.default_rng()
        self.probs_updated = False  # Indicates whether probabilities have to be recalculated
        self.alpha = alpha  # Blend between uniform distribution (alpha = 0) and Probabilities according to rank (alpha = 1)
        self.beta = beta  # Blend between full bias correction (beta = 1) and no bias correction (beta = 0)
        self.beta_increase = (1.0 - beta) / beta_increase_steps  # Adder to reach 1 within given amoutn of steps
    
    def store_experience(self, state, action, reward, next_state, terminal):
        """
        Stores given SARS Experience in the Replay Buffer.
        Returns True if the last element has been written into memory and
        it will start over replacing the first elements at the next call.
        """
        self.states[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.next_states[self.index] = next_state
        self.terminals[self.index] = terminal
        self.errors[self.index] = 1.0 if self.__len__() == 0 else self.errors[:self.__len__()].max()  # To make it likely picked the first time

        self.probs_updated = False
        
        self.index += 1
        self.index %= self.max_len  # Replace oldest Experiences if Buffer is full

        if self.index == 0:
            self.full = True
            return True
        return False
    
    def update_experiences(self, indices, errors):
        self.errors[indices] = errors
        self.probs_updated = False

    def get_experiences(self, batch_size):
        buff_len = self.__len__()
        
        if not self.probs_updated:
            abs_errors = np.abs(self.errors[:buff_len])
            sorted_indices = abs_errors.argsort()[::-1]  # Indices from highest to lowest error
            ranks = np.arange(buff_len)[sorted_indices] + 1.0
            scaled_priorities = (1.0 / ranks)**self.alpha
            
            self.probabilities[:buff_len] = scaled_priorities / scaled_priorities.sum()
            unnormed_weights = (self.probabilities[:buff_len] * buff_len)**-self.beta
            self.weights[:buff_len] = unnormed_weights / unnormed_weights.max()

            self.beta += self.beta_increase  # Update beta
            self.beta = min(1.0, self.beta)

            self.probs_updated = True

        indices = self.rng.choice(np.arange(buff_len), batch_size, p=self.probabilities[:buff_len])

        weights = np.array([self.weights[i] for i in indices], dtype=np.float32)
        states = np.array([self.states[i] for i in indices], dtype=np.float32)
        actions = np.array([self.actions[i] for i in indices], dtype=np.int16)
        rewards = np.array([self.rewards[i] for i in indices], dtype=np.float32)
        next_states = np.array([self.next_states[i] for i in indices], dtype=np.float32)
        terminals = np.array([self.terminals[i] for i in indices], dtype=np.int8)

        return indices, weights, states, actions, rewards, next_states, terminals

    def __len__(self):
        return self.max_len if self.full else self.index

class BDQ:
    def __init__(self, state_dim, action_num, hidden_layers=(500, 500, 500), gamma=0.99, learning_rate_start=0.0005,
                 learning_rate_decay_steps=50000, learning_rate_min=0.0003, weight_decay=0.001, epsilon_start=1.0, epsilon_decay_steps=20000,
                 epsilon_min=0.1, temp_start=10, temp_decay_steps=20000, temp_min=0.1, buffer_size_min=200,
                 buffer_size_max=50000, batch_size=50, replays=1, tau=0.01, alpha=0.6, beta=0.1, beta_increase_steps=20000, device='cpu'):

        self.state_dim = state_dim
        self.action_num = action_num
        self.hidden_layers = hidden_layers
        layers = (state_dim, *hidden_layers, action_num)  # 3 hidden layers of sizes 500, 500 and 500
        self.q_net = Network(layers, device).to(device)
        self.target_q_net = Network(layers, device).to(device)
        self._update_target(1.0)  # Fully copy Online Net weights to Target Net

        self.optimizer = torch.optim.RMSprop(self.q_net.parameters(),  # Using RMSProp because it's more stable than Adam
                                             lr=learning_rate_start, weight_decay=weight_decay)
        self.weight_decay = weight_decay
        self.loss_function = nn.HuberLoss()

        self.buffer = ReplayBuffer(buffer_size_max, state_dim, alpha, beta, beta_increase_steps)
        self.buffer_size_max = buffer_size_max
        self.buffer_size_min = buffer_size_min
        self.batch_size = batch_size
        self.replays = replays  # On how many batches it should train after each step
        self.alpha = alpha
        self.beta = beta
        self.beta_increase_steps = beta_increase_steps

        self.gamma = gamma  # Reward discount rate

        # Linearly decay Learning Rate and Epsilon from start to min in a given amount of steps
        self.learning_rate = learning_rate_start
        self.learning_rate_start = learning_rate_start
        self.learning_rate_decay = (learning_rate_start - learning_rate_min) / learning_rate_decay_steps
        self.learning_rate_min = learning_rate_min

        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_decay = (epsilon_start - epsilon_min) / epsilon_decay_steps
        self.epsilon_min = epsilon_min

        # Temperature for Softmax
        self.temp = temp_start
        self.temp_start = temp_start
        # Decay rate is the base which leads to decaying exponentially from start to min in given steps
        self.temp_decay_rate = (temp_min/temp_start)**(1/temp_decay_steps)
        self.temp_min = temp_min

        self.tau = tau  # Mixing parameter for polyak averaging

        self.device = device

        self.rng = np.random.default_rng()
    
    def reset(self):
        """
        Reset object to its initial state if you want to do multiple training passes with it
        """
        layers = (self.state_dim, *(self.hidden_layers), self.action_num)  # 3 hidden layers of sizes 500, 500 and 500
        self.q_net = Network(layers, self.device).to(self.device)
        self.target_q_net = Network(layers, self.device).to(self.device)
        self._update_target(1.0)  # Fully copy Online Net weights to Target Net

        self.optimizer = torch.optim.RMSprop(self.q_net.parameters(), lr=self.learning_rate_start, weight_decay=self.weight_decay)
        self.loss_function = nn.HuberLoss()

        self.buffer = ReplayBuffer(self.buffer_size_max, self.state_dim, self.alpha, self.beta, self.beta_increase_steps)

        self.learning_rate = self.learning_rate_start
        self.epsilon = self.epsilon_start
        self.temp = self.temp_start
    
    def act_e_greedy(self, state):
        """
        Decides on action based on current state using epsilon-greedy Policy.
        """
        with torch.no_grad():
            state = tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
            Q = self.q_net(state).squeeze()
            greedy_action = Q.argmax().item()

        if self.rng.random() < self.epsilon:
            action = self.rng.integers(self.action_num)  # Random
        else:
            action = greedy_action  # Greedy
        
        is_greedy = action == greedy_action

        return action, is_greedy
    
    def act_softmax(self, state):
        """
        Transform Value function to Softmax probability distribution and sample from it randomly.
        """
        state = tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
        Q = self.q_net(state).detach().squeeze()
        temp_Q = Q / self.temp
        norm_Q = temp_Q - temp_Q.max().expand_as(temp_Q)  # Normalize for numerical stability

        pi = norm_Q.exp() / norm_Q.exp().sum()
        pi = pi.cpu().numpy()

        action = self.rng.choice(np.arange(len(pi)), size=1, p=pi).item()
        is_greedy = action == Q.argmax().item()

        return action, is_greedy

    def act_greedily(self, state):
        """
        Decides on action based on current state using greedy Policy.
        """
        with torch.no_grad():
            state = tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
            Q = self.q_net(state).squeeze()
            action = Q.argmax().item()
        return action, True
    
    def experience(self, state, action, reward, next_state, terminal):
        """
        Takes experience and stores it for replay.
        """
        if self.buffer.store_experience(state, action, reward, next_state, terminal):
            self.epsilon = self.epsilon_start  # Reset exploration rate when Replay Buffer is full to always have
            self.temp = self.temp_start        # negative Experiences in storage. Will prevent catastrophic forgetting
    
    def train(self):
        """
        Train Q-Network on a batch from the replay buffer.
        """
        if len(self.buffer) < self.buffer_size_min:
            return  # Dont train until Replay Buffer has collected a certain number of initial experiences

        for _ in range(self.replays):
            indices, weights, states, actions, rewards, next_states, terminals = self.buffer.get_experiences(self.batch_size)

            weights = tensor(weights, device=self.device, dtype=torch.float32)
            states = tensor(states, device=self.device, dtype=torch.float32)
            rewards = tensor(rewards, device=self.device, dtype=torch.float32)
            actions = tensor(actions, device=self.device, dtype=torch.int64)
            next_states = tensor(next_states, device=self.device, dtype=torch.float32)
            terminals = tensor(terminals, device=self.device, dtype=torch.int8)

            with torch.no_grad():
                max_actions = self.q_net(next_states).argmax(1).unsqueeze(1)
                max_action_vals = self.target_q_net(next_states).gather(1, max_actions).squeeze()

            targets = rewards + self.gamma * max_action_vals * (1 - terminals)
            predictions = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze()
            loss = self.loss_function(weights*predictions, weights*targets)  # Huber Loss with bias correction using weights

            errors = (targets - predictions).detach().cpu().numpy()
            self.buffer.update_experiences(indices, errors)  # Update replay buffer's temporal difference errors

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self._update_parameters()
        self._update_target(self.tau)
    
    def save_net(self, path):
        torch.save(self.q_net.state_dict(), path)
    
    def load_net(self, path):
        self.q_net.load_state_dict(torch.load(path))
        self._update_target(1.0)  # Also load weights into target net
    
    def _update_target(self, tau):
        """
        Update Target Network by blending Target und Online Network weights using the factor tau.
        A tau of 1 just copies the whole online network over to the target network
        """
        for param, target_param in zip(self.q_net.parameters(), self.target_q_net.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
    
    def _update_parameters(self):
        """
        Decays parameters like learning rate and epsilon one step
        """
        self.learning_rate -= self.learning_rate_decay
        self.learning_rate = max(self.learning_rate, self.learning_rate_min)

        self.optimizer.param_groups[0]['lr'] = self.learning_rate

        self.epsilon -= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.epsilon_min)

        self.temp *= self.temp_decay_rate
        self.temp = max(self.temp, self.temp_min)