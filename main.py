from env import Env
from agent import DQNagent
import torch
import os
from torch.utils.tensorboard import SummaryWriter
import argparse
import numpy as np


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=1, type=int) # 1 表示训练，0表示只进行eval
    parser.add_argument("--train_eps", default=100, type=int)
    parser.add_argument("--train_steps", default=100, type=int)
    parser.add_argument("--gamma", default=0.99, type=float)  # q-learning中的gamma
    parser.add_argument("--epsilon_start", default=0.95, type=float)  # 基于贪心选择action对应的参数epsilon
    parser.add_argument("--epsilon_end", default=0.01, type=float)
    parser.add_argument("--epsilon_decay", default=1000, type=float)
    parser.add_argument("--policy_lr", default=0.01, type=float)
    parser.add_argument("--memory_capacity", default=1000, type=int, help="capacity of Replay Memory") 
    parser.add_argument("--batch_size", default=32, type=int, help="batch size of memory sampling")
    parser.add_argument("--target_update", default=2, type=int, help="when(every default 2 eisodes) to update target net ") # 更新频率
    parser.add_argument("--eval_eps", default=100, type=int)  # 测试的最大episode数目
    parser.add_argument("--eval_steps", default=300, type=int) 
    config = parser.parse_args()
    return config

def save_results(rewards,moving_average_rewards,ep_steps,tag='train',result_path='./result/'):
    '''保存reward等结果
    '''
    if not os.path.exists(result_path): # 检测是否存在文件夹
        os.mkdir(result_path)
    np.save(result_path+'rewards_'+tag+'.npy', rewards)
    np.save(result_path+'moving_average_rewards_'+tag+'.npy', moving_average_rewards)
    np.save(result_path+'steps_'+tag+'.npy',ep_steps )
    print('results saved!')

def save_model(agent,model_path='./saved_model/'):
    if not os.path.exists(model_path): # 检测是否存在文件夹
        os.mkdir(model_path)
    agent.save_model(model_path+'checkpoint.pth')
    print('model saved！')


def train(cfg):
    
    print("Start to train !\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = Env()
    n_state = 5
    n_action = 5
    agent = DQNagent(n_state=n_state, n_action=n_action, gamma=cfg.gamma, epsilon_start=cfg.epsilon_start, epsilon_end=cfg.epsilon_end, 
                    epsilon_decay=cfg.epsilon_decay, policy_lr=cfg.policy_lr, memory_capacity=cfg.memory_capacity, batch_size=cfg.batch_size, device=device)
    # tensorboard参数
    rewards = []
    moving_average_rewards = []
    ep_steps = []
    log_dir = "./logs/train"
    writer = SummaryWriter(log_dir)
    for i_episode in range(1, cfg.train_eps+1):
        """ env.uav_rect.left = 0
        env.uav_rect.top = 0
        # state为uav中点到五个目标的距离
        state = [env.uav_rect.midbottom[0]-50, env.uav_rect.midbottom[0]-250, env.uav_rect.midbottom[0]-450, env.uav_rect.midbottom[0]-650, env.uav_rect.midbottom[0]-850] """
        state = env.reset()
        ep_reward = 0
        for i_step in range(1, cfg.train_steps+1):
            env.render()
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            ep_reward += reward
            agent.memory.push(state, action, reward, next_state, done) # 将state等这些transition存入memory
            state = next_state
            agent.update()
            
            if done:
                break
        # 更新target network，复制DQN中的所有weights and biases
        if i_episode % cfg.target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        print("Episode:", i_episode, " Reward:", ep_reward, "n_steps:", i_step, "done:", done, "Explore:%.2f" % agent.epsilon)
        ep_steps.append(i_step)
        rewards.append(ep_reward)
        # 计算滑动窗口的reward
        if i_episode == 1:
            moving_average_rewards.append(ep_reward)
        else:
            moving_average_rewards.append(0.9*moving_average_rewards[-1]+0.1*ep_reward)
        writer.add_scalars('reward', {'raw':rewards[-1], 'moving_average':moving_average_rewards[-1]}, i_episode)
        writer.add_scalar('steps_of_each_episode', ep_steps[-1], i_episode)
    writer.close()
    # 保存模型
    save_model(agent,model_path = './saved_model/')
    # 存储reward等相关结果
    save_results(rewards,moving_average_rewards,ep_steps,tag='train',result_path='./result/')
    print('Complete training！')

def eval(cfg, saved_model_path = './saved_model/'):
    
    print("Start to eval !\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = Env()
    n_state = 5
    n_action = 5
    agent = DQNagent(n_state=n_state, n_action=n_action, gamma=cfg.gamma, epsilon_start=cfg.epsilon_start, epsilon_end=cfg.epsilon_end, 
                    epsilon_decay=cfg.epsilon_decay, policy_lr=cfg.policy_lr, memory_capacity=cfg.memory_capacity, batch_size=cfg.batch_size, device=device)
    agent.load_model(saved_model_path+'checkpoint.pth')
    # tensorboard参数
    rewards = []
    moving_average_rewards = []
    ep_steps = []
    log_dir = "./logs/eval"
    writer = SummaryWriter(log_dir)
    for i_episode in range(1, cfg.eval_eps+1):
        """ env.uav_rect.left = 0
        env.uav_rect.top = 0
        # state为uav中点到五个目标的距离
        state = [env.uav_rect.midbottom[0]-50, env.uav_rect.midbottom[0]-250, env.uav_rect.midbottom[0]-450, env.uav_rect.midbottom[0]-650, env.uav_rect.midbottom[0]-850] """
        state = env.reset()
        ep_reward = 0
        for i_step in range(1, cfg.eval_steps+1):
            
            action = agent.choose_action(state, train=False)
            next_state, reward, done = env.step(action)
            ep_reward += reward
            # agent.memory.push(state, action, reward, next_state, done) # 将state等这些transition存入memory
            state = next_state
            # agent.update()
            env.render()
            if done:
                break
        """ # 更新target network，复制DQN中的所有weights and biases
        if i_episode % cfg.target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict()) """
        print("Episode:", i_episode, " Reward:", ep_reward, "n_steps:", i_step, "done:", done)
        ep_steps.append(i_step)
        rewards.append(ep_reward)
        # 计算滑动窗口的reward
        if i_episode == 1:
            moving_average_rewards.append(ep_reward)
        else:
            moving_average_rewards.append(0.9*moving_average_rewards[-1]+0.1*ep_reward)
        writer.add_scalars('reward', {'raw':rewards[-1], 'moving_average':moving_average_rewards[-1]}, i_episode)
        writer.add_scalar('steps_of_each_episode', ep_steps[-1], i_episode)
    writer.close()
    
    # 存储reward等相关结果
    save_results(rewards,moving_average_rewards,ep_steps,tag='eval',result_path='./result/')
    print('Complete evaling！')




if __name__ == "__main__":

    cfg = get_args()
    # 训练
    if cfg.train:
        train(cfg)
        eval(cfg)
    else:
        # 测试
        model_path = "./saved_model/"
        eval(cfg, saved_model_path=model_path)