import copy
from collections import deque
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optimizers


class DQN(nn.Module):
    '''
    Simple Deep Q-Network for CartPole
    '''
    def __init__(self,
                 device='cpu'):
        super().__init__()
        self.device = device

        self.original = Network().to(device)
        self.target = Network().to(device)

    def forward(self, x):
        return self.original(x)

    def q_original(self, x):
        return self.forward(x)

    def q_target(self, x):
        return self.target(x)

    def copy_original(self):
        self.target = copy.deepcopy(self.original)


class Network(nn.Module):
    def __init__(self,
                 device='cpu'):
        super().__init__()
        self.device = device

        self.l1 = nn.Linear(4, 16)
        self.l2 = nn.Linear(16, 32)
        self.l3 = nn.Linear(32, 16)
        self.l4 = nn.Linear(16, 2)

    def forward(self, x):
        x = self.l1(x)
        x = torch.relu(x)
        x = self.l2(x)
        x = torch.relu(x)
        x = self.l3(x)
        x = torch.relu(x)
        y = self.l4(x)

        return y


class ReplayMemory(object):
    def __init__(self,
                 memory_size=50000,
                 device='cpu'):
        self.device = device
        self.memory_size = memory_size
        self.memories = deque([], maxlen=memory_size)

    def append(self, memory):
        self.memories.append(memory)

    def sample(self, batch_size=128):
        indices = \
            np.random.permutation(range(len(self.memories)))[:batch_size]\
            .tolist()

        state = np.array([self.memories[i].state for i in indices])
        action = np.array([self.memories[i].action for i in indices])
        next_state = \
            np.array([self.memories[i].next_state for i in indices])
        reward = np.array([self.memories[i].reward for i in indices])
        terminal = np.array([self.memories[i].terminal for i in indices])

        return Memory(
            torch.Tensor(state).to(self.device),
            torch.Tensor(action).to(self.device),
            torch.Tensor(next_state).to(self.device),
            torch.Tensor(reward).to(self.device),
            torch.Tensor(terminal).to(self.device),
        )


class Memory(object):
    def __init__(self,
                 state,
                 action,
                 next_state,
                 reward,
                 terminal):
        self.state = state
        self.action = action
        self.next_state = next_state
        self.reward = reward
        self.terminal = terminal


class Epsilon(object):
    def __init__(self,
                 init=1.0,
                 end=0.1,
                 steps=10000):
        self.init = init
        self.end = end
        self.steps = steps

    def __call__(self, step):
        return max(0.1,
                   self.init + (self.end - self.init) / self.steps * step)


if __name__ == '__main__':
    np.random.seed(1234)
    torch.manual_seed(1234)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def compute_loss(label, pred):
        return criterion(pred, label)

    def train_step(state, action, t):
        model.train()
        q_original = model(state)
        action = torch.eye(2)[action.long()].to(device)  # one-hot
        q = torch.sum(q_original * action, dim=1)
        loss = compute_loss(t, q)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss

    '''
    Load env
    '''
    env = gym.make('CartPole-v0')

    '''
    Build model
    '''
    model = DQN(device=device)
    criterion = nn.MSELoss()
    optimizer = optimizers.Adam(model.parameters())

    '''
    Build ReplayMemory
    '''
    initial_memory_size = 500
    replay_memory = ReplayMemory(device=device)

    step = 0
    while True:
        state = env.reset()
        terminal = False

        while not terminal:
            action = env.action_space.sample()
            next_state, reward, terminal, _ = env.step(action)
            memory = Memory(state, action, next_state, reward, int(terminal))
            replay_memory.append(memory)
            state = next_state
            step += 1

        if step >= initial_memory_size:
            break

    '''
    Train model
    '''
    n_episodes = 300
    gamma = 0.99
    step = 0
    copy_original_every = 1000
    eps = Epsilon()

    model.copy_original()
    for episode in range(n_episodes):
        state = env.reset()
        terminal = False

        rewards = 0.
        q_max = []
        while not terminal:
            s = torch.Tensor(state[None]).to(device)
            q = model.q_original(s)
            q_max.append(q.max().data.cpu().numpy())

            # epsilon-greedy
            if np.random.random() < eps(step):
                action = env.action_space.sample()
            else:
                action = torch.argmax(q).data.cpu().numpy()

            next_state, reward, terminal, _ = env.step(action)
            rewards += reward

            memory = Memory(state, action, next_state, reward, int(terminal))
            replay_memory.append(memory)

            sample = replay_memory.sample()
            q_target = model.q_target(sample.next_state)

            t = sample.reward \
                + (1 - sample.terminal) * gamma * q_target.max(-1)[0]

            train_step(sample.state, sample.action, t)

            state = next_state
            env.render()

            if (step + 1) % copy_original_every == 0:
                model.copy_original()

            step += 1

        template = 'Episode: {}, Reward: {}, Qmax: {:.3f}'
        print(template.format(
            episode+1,
            rewards,
            np.mean(q_max)
        ))

    env.close()
