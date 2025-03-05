# -*- coding: utf-8 -*-

import random
import numpy as np
import collections
import torch
import os
from tqdm import tqdm

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)
        return {
            'states': np.array(states),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'next_states': np.array(next_states),
            'dones': np.array(dones)
        }

    def size(self): 
        return len(self.buffer)

def plot_path_and_obstacles(env, dir, episode_num):
    import matplotlib.pyplot as plt
    
    path = np.array(env.path)
    obstacles = env.insideBuildingGrid  # 障碍物数据
    obstacle_points = np.argwhere(obstacles)

    # 创建二维绘图窗口
    fig, ax = plt.subplots(figsize=(10, 10))

    # 绘制障碍物点
    ax.scatter(obstacle_points[:, 1], obstacle_points[:, 0], c='red', marker='s', s=20, label='Obstacles', alpha=0.15)

    # 绘制无人机路径
    if path.ndim == 2 and path.shape[1] == 2:
        ax.plot(path[:, 1], path[:, 0], color='blue', linewidth=1, label='Drone Path')
        ax.scatter(path[:, 1], path[:, 0], c='blue', marker='o', s=5)
        # 绘制路径起点和终点
        ax.scatter(path[0, 1], path[0, 0], c='yellow', marker='o', s=100, label='Start')
        ax.scatter(path[-1, 1], path[-1, 0], c='purple', marker='x', s=100, label='End')
    elif path.ndim == 1 and path.size == 2:
        ax.scatter(path[1], path[0], c='blue', marker='o', s=5, label='Drone Position')

    # 绘制源点
    ax.scatter(env.source[1], env.source[0], c='green', marker='*', s=100, label='source')

    shape = env.powerGrid.shape
    ax.set_xlim(0, shape[1])
    ax.set_ylim(0, shape[0])
    ax.set_xlabel('Y')
    ax.set_ylabel('X')
    ax.set_title(f'Drone Path and Obstacles in Episode {episode_num}')
    ax.legend()
    plt.savefig(os.path.join(dir, f'2d_path_{episode_num}.png'))
    plt.close()

# 保存模型权重
def save_model(agent, filename):
    torch.save(agent.q_net.state_dict(), filename)

# 加载模型权重
def load_model(agent, filename):
    agent.q_net.load_state_dict(torch.load(filename, weights_only=True))
    agent.q_net.eval()

# 评估模型
def evaluate_model(mode, agent, env, idx, num_episodes):
    print("Evaluating model...")
    success_num = 0
    return_list = []
    with tqdm(total=num_episodes, desc="Evaluating") as pbar:
        for i in range(num_episodes):
            episode_return = 0
            state = env.reset()
            done = False

            while not done:
                action = agent.take_action(state)
                next_state, reward, done = env.step(action)
                state = next_state
                episode_return += reward

                if done and reward == 2048:
                    success_num += 1
            return_list.append(episode_return)
            if i % 50 == 0:
                # 调用绘图函数
                if mode == "train":
                    plot_path_and_obstacles(env, "dqn_pics/paths/test/2d", f'eval_{idx+1}_{i+1}')
                elif mode == "eval":
                    plot_path_and_obstacles(env, "dqn_pics/paths/test/2d/complex", f'eval_{idx+1}_{i+1}')
                # 更新进度条
                pbar.set_postfix({"episode": f"{i+1}/{num_episodes}", "return": f"{episode_return:.3f}", "success": success_num})
            pbar.update(1)
    success_rate = success_num / num_episodes
    return return_list, success_rate


class Environment:
    def __init__(self, powerGrid, insideBuildingGrid, esdf, gradLat, gradLon, gradESDFLat, gradESDFLon, source, max_step_num):
        self.powerGrid = powerGrid
        self.insideBuildingGrid = insideBuildingGrid
        self.esdf = esdf
        self.grad_x = gradLat
        self.grad_y = gradLon
        self.gradESDF_x = gradESDFLat
        self.gradESDF_y = gradESDFLon
        self.source = source
        self.max_step_num = max_step_num
        self.x, self.y = powerGrid.shape
        self.reset()

    def reset(self):
        safe_margin = 10

        self.drone_position = (
            np.random.randint(safe_margin, self.x - safe_margin),
            np.random.randint(safe_margin, self.y - safe_margin)
        )
        # while (self.insideBuildingGrid[self.drone_position[0], self.drone_position[1], self.drone_position[2]] or self.is_around_source(self.drone_position[0], self.drone_position[1], self.drone_position[2], 0.1 * (1 + 2 * np.exp(-self.episode_num / 5000)) + 0.1)):
        #     self.drone_position = np.random.randint(0, self.x), np.random.randint(0, self.y), np.random.randint(0, self.z)
#             #   or self.is_far_from_source(self.drone_position[0], self.drone_position[1], 0.5)\
        while self.insideBuildingGrid[self.drone_position[0], self.drone_position[1]]\
              or self.is_around_source(self.drone_position[0], self.drone_position[1], 0.1)\
              or not self.check_path_feasibility(self.drone_position[0], self.drone_position[1]):
            self.drone_position = (
                np.random.randint(safe_margin, self.x - safe_margin),
                np.random.randint(safe_margin, self.y - safe_margin),
            )    
        self.path = []
        self.steps = 0
        return self.get_state()
    

    def check_path_feasibility(self, start_x, start_y):
        """使用BFS判断从(start_x, start_y)到self.source是否存在可行路径"""
        queue = collections.deque([(start_x, start_y)])
        visited = set([(start_x, start_y)])
        while queue:
            cx, cy = queue.popleft()
            # 到达源点附近，返回True
            # if (cx, cy) == (self.source[0], self.source[1]):
            if self.is_around_source(cx, cy, 0.1):
                return True
            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                nx, ny = cx + dx, cy + dy
                if not self.is_outside(nx, ny)\
                    and (nx, ny) not in visited \
                    and not self.insideBuildingGrid[nx, ny]:
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        return False

    # def is_line_of_sight_clear(self, x, y):
    #     """检查从 (x, y) 到源点的直线上是否有障碍物"""
    #     x0, y0 = x, y
    #     x1, y1 = self.source

    #     # 使用 Bresenham 算法检查直线上是否有障碍物
    #     dx = abs(x1 - x0)
    #     dy = abs(y1 - y0)
    #     sx = 1 if x0 < x1 else -1
    #     sy = 1 if y0 < y1 else -1
    #     err = dx - dy

    #     while True:
    #         if self.insideBuildingGrid[x0, y0]:
    #             return False
    #         if x0 == x1 and y0 == y1:
    #             break
    #         e2 = err * 2
    #         if e2 > -dy:
    #             err -= dy
    #             x0 += sx
    #         if e2 < dx:
    #             err += dx
    #             y0 += sy

    #     return True

    def is_outside(self, x, y):
        return x < 0 or x >= self.x or y < 0 or y >= self.y
    
    def is_around_source(self, x, y, weight):
        distance_squre = (x - self.source[0]) ** 2 + (y - self.source[1]) ** 2
        threshold_distance = (weight * self.x) ** 2
        if distance_squre <= threshold_distance:
            return True
        else:
            return False
    
    def is_far_from_source(self, x, y, weight):
        distance_squre = (x - self.source[0]) ** 2 + (y - self.source[1]) ** 2
        threshold_distance = (weight * self.x) ** 2
        if distance_squre > threshold_distance:
            return True
        else:
            return False
    
    def get_state(self):
        x, y = self.drone_position

        # 定义提取的区域范围，window_size 为奇数
        window_size = 9
        half_window = window_size // 2

        x_min = max(0, x - half_window)
        x_max = min(self.x, x + half_window + 1)
        y_min = max(0, y - half_window)
        y_max = min(self.y, y + half_window + 1)

        # 初始化状态张量，形状为 [通道数, 高度, 宽度]
        power_square = np.zeros((window_size, window_size))
        grad_x_square = np.zeros((window_size, window_size))
        grad_y_square = np.zeros((window_size, window_size))
        esdf_square = np.zeros((window_size, window_size))
        grad_esdf_x_square = np.zeros((window_size, window_size))
        grad_esdf_y_square = np.zeros((window_size, window_size))
        # steps_square = np.zeros((window_size, window_size))

        # 计算在状态张量中的起始和结束位置
        x_start = half_window - (x - x_min)
        x_end = x_start + (x_max - x_min)
        y_start = half_window - (y - y_min)
        y_end = y_start + (y_max - y_min)

        power_square[x_start:x_end, y_start:y_end] = self.powerGrid[x_min:x_max, y_min:y_max]
        grad_x_square[x_start:x_end, y_start:y_end] = self.grad_x[x_min:x_max, y_min:y_max]
        grad_y_square[x_start:x_end, y_start:y_end] = self.grad_y[x_min:x_max, y_min:y_max]
        esdf_square[x_start:x_end, y_start:y_end] = self.esdf[x_min:x_max, y_min:y_max]
        grad_esdf_x_square[x_start:x_end, y_start:y_end] = self.gradESDF_x[x_min:x_max, y_min:y_max]
        grad_esdf_y_square[x_start:x_end, y_start:y_end] = self.gradESDF_y[x_min:x_max, y_min:y_max]

        # # 添加步数信息，归一化为 [0,1]
        # normalized_steps = self.steps / self.max_step_num
        # steps_square.fill(normalized_steps)

        # 将6个通道堆叠起来，形成形状为 [6, 10, 10] 的状态张量
        state = np.stack([
            power_square,
            grad_x_square,
            grad_y_square,
            esdf_square,
            grad_esdf_x_square,
            grad_esdf_y_square,
            # steps_square,
        ], axis=0)

        return state
    
    def get_power_reward(self, x, y):
        """计算以(x,y)为中心的区域功率平均值"""
        size = 3
        x_min = max(0, x - size)
        x_max = min(self.x, x + size + 1)
        y_min = max(0, y - size)
        y_max = min(self.y, y + size + 1)
        return np.mean(self.powerGrid[x_min:x_max, y_min:y_max])

    def step(self, action):
        base_reward = -25

        directions = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1)
        ]

        dx, dy = directions[action]
        new_dx, new_dy = dx, dy
        new_x, new_y = self.drone_position[0] + dx, self.drone_position[1] + dy

        self.steps += 1

        # 碰到边界，给予负奖励，终止
        if self.is_outside(new_x, new_y):
            reward = -128
            return self.get_state(), reward, True
        
        # 碰到障碍物，给予负奖励，终止
        if self.insideBuildingGrid[new_x, new_y]:
            reward = -128
            return self.get_state(), reward, True

        # # 碰到边界，给予负奖励，随机挑选一个其他action执行，继续
        # if self.is_outside(new_x, new_y):
        #     while (new_dx, new_dy) == (dx, dy) or self.is_outside(new_x, new_y) or self.insideBuildingGrid[new_x, new_y]:
        #         new_dx, new_dy = random.choice(directions)
        #         new_x, new_y = self.drone_position[0] + new_dx, self.drone_position[1] + new_dy
        #     reward = -50
        #     self.drone_position = (new_x, new_y)
        #     self.path.append(self.drone_position)
        #     return self.get_state(), reward, False
        
        # # 碰到障碍物，给予负奖励，随机挑选一个其他action执行，继续
        # if self.insideBuildingGrid[new_x, new_y]:
        #     while (new_dx, new_dy) == (dx, dy) or self.is_outside(new_x, new_y) or self.insideBuildingGrid[new_x, new_y]:
        #         new_dx, new_dy = random.choice(directions)
        #         new_x, new_y = self.drone_position[0] + new_dx, self.drone_position[1] + new_dy
        #     reward = -50
        #     self.drone_position = (new_x, new_y)
        #     self.path.append(self.drone_position)
        #     return self.get_state(), reward, False

        # 到达源点附近且与源点的连线上没有障碍物，给予高奖励，结束
        if self.is_around_source(new_x, new_y, weight = 0.05):
        # if self.is_around_source(new_x, new_y, weight = 0.15) and self.is_line_of_sight_clear(new_x, new_y):
            self.drone_position = (new_x, new_y)
            self.path.append(self.drone_position)
            reward = 2048
            return self.get_state(), reward, True

        prev_dist = np.linalg.norm(np.array(self.drone_position) - self.source)
        new_dist = np.linalg.norm(np.array([new_x, new_y]) - self.source)
        dist_reward = (prev_dist - new_dist) * 20

        # 靠近源点时功率衰减速度较快，这部分较大，远离源点时功率衰减速度较慢，这部分较小
        prev_power = self.get_power_reward(self.drone_position[0], self.drone_position[1])
        new_power = self.get_power_reward(new_x, new_y)
        power_reward = (new_power - prev_power) * 20

        reward = base_reward + dist_reward + power_reward
        # print(f"dist_reward: {dist_reward}, power reward: {power_reward} reward: {reward}")

        # 检查是否超过最大步数
        if self.steps >= self.max_step_num: # 超过最大步数，结束
            # reward = -10240
            self.drone_position = (new_x, new_y)
            self.path.append(self.drone_position)
            return self.get_state(), reward, True
        else: # 走一步，继续
            self.drone_position = (new_x, new_y)
            self.path.append(self.drone_position)
            return self.get_state(), reward, False

