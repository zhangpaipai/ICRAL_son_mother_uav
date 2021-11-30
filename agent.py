import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
import random
from memory import ReplayBuffer

class FCN(nn.Module):
    def __init__(self, n_state = 5, n_action = 5, hidden_dim = 128):
        super(FCN, self).__init__()
        self.fc1 = nn.Linear(n_state, hidden_dim) # 输入层
        self.fc2 = nn.Linear(hidden_dim,hidden_dim) # 隐藏层
        self.fc3 = nn.Linear(hidden_dim, n_action) # 输出层

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DQNagent:
    def __init__(self, n_state, n_action, gamma=0.99, epsilon_start=0.95, epsilon_end=0.05, epsilon_decay=200, memory_capacity=10000, policy_lr=0.01, batch_size=128, device='cpu'):
        
        self.device = device
        self.n_state = n_state
        self.n_action = n_action
        self.policy_net = FCN(n_state, n_action, hidden_dim = 128).to(self.device)
        self.target_net = FCN(n_state, n_action, hidden_dim = 128).to(self.device)
        # e-greedy策略相关参数
        self.epsilon = 0
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.actions_count = 0 # 用于epsilon的衰减计数
        # update参数
        self.batch_size = batch_size
        self.gamma = gamma
        self.loss = 0
        # 可查parameters()与state_dict()的区别，前者require_grad=True
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        # ReplayBuffer参数
        self.memory = ReplayBuffer(memory_capacity)


    def update(self):
        if len(self.memory) < self.batch_size:
            return
        # 从memory中随机采样transition
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(self.batch_size)
        
        state_batch = torch.tensor(state_batch, device=self.device, dtype=torch.float)
        action_batch = torch.tensor(action_batch, device=self.device).unsqueeze(1)  # 例如tensor([[1],...,[0]])
        reward_batch = torch.tensor(reward_batch, device=self.device, dtype=torch.float)  # tensor([1., 1.,...,1])
        next_state_batch = torch.tensor(next_state_batch, device=self.device, dtype=torch.float)
        done_batch = torch.tensor(np.float32(done_batch), device=self.device).unsqueeze(1)  # 将bool转为float然后转为张量
        '''计算当前(s_t,a)对应的Q(s_t, a)'''
        '''torch.gather:对于a=torch.Tensor([[1,2],[3,4]]),那么a.gather(1,torch.Tensor([[0],[1]]))=torch.Tensor([[1],[3]])'''
        q_values = self.policy_net(state_batch).gather(dim=1, index=action_batch)  # 等价于self.forward
        # 计算所有next states的V(s_{t+1})，即通过target_net中选取reward最大的对应states
        next_state_values = self.target_net(next_state_batch).max(1)[0].detach()  # 比如tensor([ 0.0060, -0.0171,...,])
        # 计算 expected_q_value
        # 对于终止状态，此时done_batch[0]=1, 对应的expected_q_value等于reward
        expected_q_values = reward_batch + self.gamma * next_state_values * (1-done_batch[0])
        # 计算均方误差loss
        self.loss = nn.MSELoss()(q_values, expected_q_values.unsqueeze(1))  # 计算 均方误差loss
        # 优化模型
        self.optimizer.zero_grad()  # zero_grad清除上一步所有旧的gradients from the last step
        # loss.backward()使用backpropagation计算loss相对于所有parameters(需要gradients)的微分
        self.loss.backward()
        for param in self.policy_net.parameters():  # clip防止梯度爆炸
            param.grad.data.clamp_(-1, 1)
            
        self.optimizer.step()  # 更新模型

    def choose_action(self, state, train=True):
        # 训练时
        if train:
            self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(-1. * self.actions_count / self.epsilon_decay)
            self.actions_count += 1
            if random.random() > self.epsilon:
                with torch.no_grad(): # 取消保存梯度
                    # 先转为张量便于丢给神经网络,state元素数据原本为float64
                    # 注意state=torch.tensor(state).unsqueeze(0)跟state=torch.tensor([state])等价
                    state = torch.tensor([state], device=self.device, dtype=torch.float32)
                    # 每个action对应的Q值向量
                    q_value = self.target_net(state)
                    action = q_value.max(dim=1)[1].item()
            else:
                action = random.randrange(self.n_action)
            return action
        # 测试时
        else:
            with torch.no_grad(): # 取消保存梯度
                # 先转为张量便于丢给神经网络,state元素数据原本为float64
                # 注意state=torch.tensor(state).unsqueeze(0)跟state=torch.tensor([state])等价
                state = torch.tensor([state], device=self.device, dtype=torch.float32)
                # 每个action对应的Q值向量
                q_value = self.target_net(state)
                action = q_value.max(dim=1)[1].item()
            return action

    def save_model(self, path):
        torch.save(self.target_net.state_dict(), path)
        
    def load_model(self, path):
        self.target_net.load_state_dict(torch.load(path))
