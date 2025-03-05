# -*- coding: utf-8 -*-

import random
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import rl_utils
import os
from scipy.io import loadmat

from twod_agent import DQN
from twod_utils import ReplayBuffer, Environment, plot_path_and_obstacles, save_model, load_model, evaluate_model

# 加噪声
# 到达源点周围区域就给予奖励
# 初始点设置在周围区域

# 后续工作
# 模型泛用性测试，更换终点位置
# 寻找类似工作进行对比
# 优化模型，提高效率
# 写论文

if __name__ == "__main__":
    # 设置环境变量以指定使用的GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    mode = "train"
    # mode = "train_complex"
    # mode = "eval"

    # 读取 .mat 文件
    data_list = []
    if mode == "train":
        mat_folder = "./outputData"
    elif mode == "eval" or mode == "train_complex":
        mat_folder = "./outputDataComplex"

    # 获取所有 .mat 文件并排序
    mat_files = sorted([f for f in os.listdir(mat_folder) if f.endswith('.mat')])

    for filename in mat_files:
        file_path = os.path.join(mat_folder, filename)
        data = loadmat(file_path)
        data_list.append((filename, data))

    # for filename in ['gridData1.mat', 'gridData2.mat', 'gridData3.mat', 'gridData4.mat', 'gridData5.mat', 'gridData6.mat', 'gridData7.mat', 'gridData8.mat', 'gridData9.mat', 'gridData10.mat', 'gridData11.mat', 'gridData12.mat', 'gridData13.mat', 'gridData14.mat']:
    #     data = loadmat(os.path.join(mat_folder, filename))
    #     data_list.append(data)

    # def normalize(data):
    #     min_val = np.min(data)
    #     max_val = np.max(data)
    #     if min_val == max_val:
    #         print(f"Min value equals to max value: {min_val}")
    #         return data  # 或者其他默认值
    #     return (data - min_val) / (max_val - min_val) * 2 - 1

    # env_list = []
    # inf_cnt = 0
    # for idx, (filename, data) in enumerate(data_list):
    #     powerGrid = data['powerGrid']
    #     insideBuildingGrid = data['insideBuildingGrid']
    #     gradLat = data['gradLat']
    #     gradLon = data['gradLon']
    #     # 如果esdf中的数据全都是inf，则将esdf设置为全0
    #     if np.all(np.isinf(data['ESDF'])):
    #         print(f"ESDF data in {filename} is all inf")
    #         inf_cnt += 1
    #         esdf = np.zeros_like(data['ESDF'])
    #         gradESDFLat = np.zeros_like(data['gradESDFLat'])
    #         gradESDFLon = np.zeros_like(data['gradESDFLon'])
    #     else: # 否则读取esdf数据
    #         esdf = data['ESDF']
    #         gradESDFLat = data['gradESDFLat']
    #         gradESDFLon = data['gradESDFLon']
        
    #     shape = powerGrid.shape
    #     # print(f"Power Grid Shape: {shape}")
    #     max_index = (shape[0] // 2, shape[1] // 2)
    #     # max_step_num = shape[0] * shape[1]
    #     max_step_num = 500

    #     # 避免0的出现
    #     powerGrid[powerGrid == 0] = -50

    #     # # 归一化到[-1, 1]
    #     # powerGrid = normalize(powerGrid)
    #     # gradLat = normalize(gradLat)
    #     # gradLon = normalize(gradLon)
    #     # esdf = normalize(esdf)
    #     # gradESDFLat = normalize(gradESDFLat)
    #     # gradESDFLon = normalize(gradESDFLon)

    #     # assert max_index_2d == np.unravel_index(np.argmax(powerGrid_2d, axis=None), powerGrid_2d.shape), "Max index error"
    #     # x = np.unravel_index(np.argmax(powerGrid_2d, axis=None), powerGrid_2d.shape)

    #     # print(f"env:{idx+1}, Power Grid 2D Shape: {shape}, Max Index 2D: {max_index}, max_step_num: {max_step_num}")

    #     env = Environment(powerGrid, insideBuildingGrid, esdf, gradLat, gradLon, gradESDFLat, gradESDFLon, max_index, max_step_num)

    #     obstacles = env.insideBuildingGrid  # 障碍物数据
    #     obstacle_points = np.argwhere(obstacles)

    #     # 创建二维绘图窗口
    #     fig, ax = plt.subplots(figsize=(10, 10))

    #     # 绘制障碍物点
    #     ax.scatter(obstacle_points[:, 1], obstacle_points[:, 0], c='red', marker='s', s=5, label='Obstacles', alpha=0.15)

    #     # 绘制源点
    #     ax.scatter(env.source[1], env.source[0], c='green', marker='*', s=100, label='source')

    #     # 设置 x 和 y 轴的范围为shape
    #     ax.set_xlim([0, shape[1]])
    #     ax.set_ylim([0, shape[0]])
    #     ax.set_xlabel('Y')
    #     ax.set_ylabel('X')
    #     ax.set_title(f'Environment 2D {idx + 1}')
    #     ax.legend()
    #     if mode == "train":
    #         plt.savefig(os.path.join("dqn_pics/env_2d/easy", f'{os.path.splitext(filename)[0]}.png'))
    #     elif mode == "eval" or mode == "train_complex":
    #         plt.savefig(os.path.join("dqn_pics/env_2d/complex", f'{os.path.splitext(filename)[0]}.png'))
    #     plt.close()

    #     env_list.append(env)
    # print("env_list number:", len(env_list), "inf_cnt:", inf_cnt)

    # # 保存环境数据
    # if mode == "train":
    #     np.save("env_list_easy.npy", env_list)
    # elif mode == "eval" or mode == "train_complex":
    #     np.save("env_list_complex.npy", env_list)
    
    # 读取环境数据
    if mode == "train":
        env_list = np.load("env_list_easy.npy", allow_pickle=True)
    elif mode == "eval" or mode == "train_complex":
        env_list = np.load("env_list_complex.npy", allow_pickle=True)
    print("env_list number:", len(env_list))

    lr = 1e-4
    num_episodes = 10000
    gamma = 0.90
    epsilon = 1.0
    # epsilon_decay = 1 - 1e-4 # 0.9999 ** 10000 = 0.36786104643297046
    epsilon_decay = 0.999
    epsilon_min = 0.01
    target_update = 500
    buffer_size = 200000
    minimal_size = 5000
    batch_size = 64
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    replay_buffer = ReplayBuffer(buffer_size)
    env_name = "Search"
    action_dim = 4 # 动作维度，前后左右
    agent = DQN(action_dim, lr, gamma, epsilon, epsilon_decay, epsilon_min, target_update, device)

    if mode == "train" or mode == "train_complex":
        # 如果是训练复杂环境，加载简单环境的模型权重
        if mode == "train_complex":
            load_model(agent, os.path.join("weights/2d", "2d_dqn_model_weights_iteration_10.pth"))
            agent.epsilon = epsilon / 2
        # ------------------- 训练模型 ----------------
        return_list = []
        for i in range(10):
            with tqdm(total=int(num_episodes / 10), desc="Iteration %d" % (i + 1)) as pbar:
                for i_episode in range(int(num_episodes / 10)):
                    env = random.choice(env_list[:len(env_list) // 2])
                    episode_return = 0
                    state = env.reset()
                    done = False

                    while not done:
                        action = agent.take_action(state)
                        next_state, reward, done = env.step(action)
                        replay_buffer.add(state, action, reward, next_state, done)
                        state = next_state
                        episode_return += reward
                        if replay_buffer.size() > minimal_size:
                            # if replay_buffer.size() == minimal_size + 1:
                            #     print("start update")
                            transitions = replay_buffer.sample(batch_size)
                            agent.update(transitions)
                    # if i >= 1:
                    agent.update_epsilon()

                    return_list.append(episode_return)

                    if (i_episode + 2) % 2 == 0:
                        pbar.set_postfix(
                            {
                                "episode": "%d" % (num_episodes // 10 * i + i_episode + 1),
                                "return": "%.3f" % np.mean(return_list[-20:]),
                            }
                        )
                        plot_path_and_obstacles(env, "dqn_pics/paths/train/2d", num_episodes // 10 * i + i_episode + 1)
                    pbar.update(1)

            if mode == "train":
                image_i = i
            elif mode == "train_complex":
                image_i = i + 10

            episodes_list = list(range(len(return_list)))
            plt.plot(episodes_list[-int(num_episodes / 10):], return_list[-int(num_episodes / 10):])
            plt.xlabel("Episodes")
            plt.ylabel("Returns")
            plt.title(f"DQN on {env_name} in iteration {image_i+1}")
            plt.savefig(os.path.join("dqn_pics/returns/2d", f"2d_returns_iteration_{image_i+1}.png"))
            plt.close()

            mv_return = rl_utils.moving_average(return_list[-int(num_episodes / 10):], 49) # 窗长必须是奇数
            plt.plot(episodes_list[-int(num_episodes / 10):], mv_return)
            plt.xlabel("Episodes")
            plt.ylabel("Returns")
            plt.title(f"DQN on {env_name} in iteration {image_i+1}")
            plt.savefig(os.path.join("dqn_pics/returns/2d", f"2d_moving_average_returns_iteration_{image_i+1}.png"))
            plt.close()

            # 保存模型权重
            save_model(agent, os.path.join("weights/2d", f"2d_dqn_model_weights_iteration_{image_i+1}.pth"))


    # ------------------- 评估模型 -------------------
    # 加载模型权重
    if mode == "eval" or mode == "train":
        load_model(agent, os.path.join("weights/2d", f"2d_dqn_model_weights_iteration_10.pth"))
    elif mode == "train_complex":
        load_model(agent, os.path.join("weights/2d", f"2d_dqn_model_weights_iteration_20.pth"))
    agent.epsilon = 0
    success_rate_list = []

    # 获取当前时间并写入
    import time
    with open('success_rate.txt', 'a') as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '\n')

    if mode == "train":
        for idx, env in enumerate(env_list[-len(env_list) // 2:]):
        # for env in env_list[-len(env_list) // 2:]:
            eval_return_list, success_rate = evaluate_model(mode, agent, env, idx, 3000)
            print(f"Evaluation {idx + 1} success rate: {success_rate}")
            success_rate_list.append(success_rate)

            # 绘制评估结果
            plt.plot(range(len(eval_return_list)), eval_return_list)
            plt.xlabel("Episodes")
            plt.ylabel("Returns")
            plt.title(f"Evaluation {idx + 1}")
            plt.savefig(os.path.join("dqn_pics/returns/2d/eval", f"2d_evaluation_returns_{idx + 1}.png"))
            plt.close()

            # 滑动平均
            mv_return = rl_utils.moving_average(eval_return_list, 99)
            plt.plot(range(len(eval_return_list)), mv_return)
            plt.xlabel("Episodes")
            plt.ylabel("Returns")
            plt.title(f"Evaluation {idx + 1}")
            plt.savefig(os.path.join("dqn_pics/returns/2d/eval", f"2d_evaluation_moving_average_returns_{idx + 1}.png"))
            plt.close()

            # 保存成功率
            with open('success_rate.txt', 'a') as f:
                f.write(f"Evaluation {idx + 1} success rate: {success_rate}\n")
    elif mode == "eval" or mode == "train_complex":
        if mode == "eval":
            for idx, env in enumerate(env_list):
                eval_return_list, success_rate = evaluate_model(mode, agent, env, idx, 3000)
                print(f"Evaluation {idx + 1} success rate: {success_rate}")
                success_rate_list.append(success_rate)

                # 绘制评估结果
                plt.plot(range(len(eval_return_list)), eval_return_list)
                plt.xlabel("Episodes")
                plt.ylabel("Returns")
                plt.title(f"Evaluation {idx + 1}")
                plt.savefig(os.path.join("dqn_pics/returns/2d/complex", f"2d_evaluation_returns_{idx + 1}.png"))
                plt.close()

                # 滑动平均
                mv_return = rl_utils.moving_average(eval_return_list, 99)
                plt.plot(range(len(eval_return_list)), mv_return)
                plt.xlabel("Episodes")
                plt.ylabel("Returns")
                plt.title(f"Evaluation {idx + 1}")
                plt.savefig(os.path.join("dqn_pics/returns/2d/complex", f"2d_evaluation_moving_average_returns_{idx + 1}.png"))
                plt.close()

                # 保存成功率
                with open('success_rate.txt', 'a') as f:
                    f.write(f"Evaluation {idx + 1} success rate: {success_rate}\n")
        elif mode == "train_complex":
            for idx, env in enumerate(env_list[-len(env_list) // 2:]):
                eval_return_list, success_rate = evaluate_model(mode, agent, env, idx, 3000)
                print(f"Evaluation {idx + 1} success rate: {success_rate}")
                success_rate_list.append(success_rate)

                # 绘制评估结果
                plt.plot(range(len(eval_return_list)), eval_return_list)
                plt.xlabel("Episodes")
                plt.ylabel("Returns")
                plt.title(f"Evaluation {idx + 1}")
                plt.savefig(os.path.join("dqn_pics/returns/2d/complex", f"2d_evaluation_returns_{idx + 1}.png"))
                plt.close()

                # 滑动平均
                mv_return = rl_utils.moving_average(eval_return_list, 99)
                plt.plot(range(len(eval_return_list)), mv_return)
                plt.xlabel("Episodes")
                plt.ylabel("Returns")
                plt.title(f"Evaluation {idx + 1}")
                plt.savefig(os.path.join("dqn_pics/returns/2d/complex", f"2d_evaluation_moving_average_returns_{idx + 1}.png"))
                plt.close()

                # 保存成功率
                with open('success_rate.txt', 'a') as f:
                    f.write(f"Evaluation {idx + 1} success rate: {success_rate}\n")
    # 求平均成功率
    success_rate_mean = np.mean(success_rate_list)
    with open('success_rate.txt', 'a') as f:
        f.write(f"Mean success rate: {success_rate_mean}\n")
