# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Qnet(nn.Module):
    def __init__(self, action_dim):
        super().__init__()

        # self.power_branch = nn.Sequential(
        #     nn.Conv2d(1, 32, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(32, 64, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.LeakyReLU(),
        #     nn.AdaptiveAvgPool2d((2,2)),
        #     nn.Flatten()
        # )

        # self.power_grad_branch = nn.Sequential(
        #     nn.Conv2d(2, 32, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(32, 64, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.LeakyReLU(),
        #     nn.AdaptiveAvgPool2d((2,2)),
        #     nn.Flatten()
        # )

        # self.esdf_branch = nn.Sequential(
        #     nn.Conv2d(1, 32, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(32, 64, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.LeakyReLU(),
        #     nn.AdaptiveAvgPool2d((2,2)),
        #     nn.Flatten()
        # )
        
        # self.esdf_grad_branch = nn.Sequential(
        #     nn.Conv2d(2, 32, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(32, 64, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.LeakyReLU(),
        #     nn.AdaptiveAvgPool2d((2,2)),
        #     nn.Flatten()
        # )

        self.conv_layers = nn.Sequential(
            nn.Conv2d(6, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten()
        )

        # Dueling DQN
        self.feature_layer = nn.Sequential(
            nn.Linear(256*2*2, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 1)  # 状态价值函数 V(s)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, action_dim),  # 优势函数 A(s, a)
        )

    # Dueling DQN
    def forward(self, state):
        # power = state[:, 0:1, :, :]
        # power_grad = state[:, 1:3, :, :]
        # esdf = state[:, 3:4:, :, :]
        # esdf_grad = state[:, 4:6, :, :]

        # power_features = self.power_branch(power)
        # power_grad_features = self.power_grad_branch(power_grad)
        # esdf_features = self.esdf_branch(esdf)
        # esdf_grad_features = self.esdf_grad_branch(esdf_grad)

        # features = torch.cat((power_features, power_grad_features, esdf_features, esdf_grad_features), dim=1)

        features = self.conv_layers(state)
        features = self.feature_layer(features)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))  # Q(s, a) = V(s) + (A(s, a) - mean(A))
        return q_values


class DQN:
    def __init__(self, action_dim, learning_rate, gamma, epsilon, epsilon_decay, epsilon_min, target_update, device):
        self.action_dim = action_dim

        self.q_net = Qnet(self.action_dim).to(device)
        self.target_q_net = Qnet(self.action_dim).to(device)

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.target_update = target_update
        self.count = 0
        self.device = device

    def take_action(self, state):  # epsilon-贪婪策略采取动作
        # 以epsilon的概率随机选择动作
        if np.random.random() < self.epsilon:
            # 生成一个 [0, self.action_dim) 之间的随机整数，表示随机选择一个动作。
            action = np.random.randint(self.action_dim)
        # 以1-epsilon的概率选择Q值最大的动作
        else:
            with torch.no_grad():
                self.q_net.eval()
                # 当调用 self.q_net(state) 时，实际上是调用了 Qnet 类的 forward 方法，计算了输出
                state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
                q_values = self.q_net(state)
                action = q_values.argmax(dim=1).item()
                self.q_net.train()
        return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).unsqueeze(1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).unsqueeze(1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)

        with torch.no_grad():
            next_q_values = self.target_q_net(next_states).max(1, keepdim=True)[0]
            q_targets = rewards + self.gamma * next_q_values * (1 - dones)

        dqn_loss = F.mse_loss(q_values, q_targets)  # 均方误差损失函数

        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())  # 更新目标网络
        self.count += 1
    
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        # self.epsilon = max(self.epsilon_min, self.epsilon - 0.0002)
