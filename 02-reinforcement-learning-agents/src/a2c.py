import time
import random
import numpy as np
import ale_py
import gymnasium as gym
import gym
import torch
import torch.nn as nn
from torch import optim
from torch.distributions import Categorical
import torch.nn.functional as F
from collections import deque
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from itertools import count
import random, pickle, os.path, math, glob

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# A2C超参数
GAMMA = 0.99
LEARNING_RATE_ACTOR = 3e-4
LEARNING_RATE_CRITIC = 1e-3
ENTROPY_COEF = 0.01
VALUE_LOSS_COEF = 0.5
MAX_GRAD_NORM = 0.5
n_episode = 20000

MODEL_STORE_PATH = 'models'
modelname = 'A2C_Breakout'

# A2C网络：包含Actor和Critic
class A2CNetwork(nn.Module):
    def __init__(self, in_channels=4, n_actions=14):
        super(A2CNetwork, self).__init__()
        # 共享特征提取层
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(7*7*64, 512)
        
        # Actor网络 - 输出动作概率
        self.actor = nn.Linear(512, n_actions)
        
        # Critic网络 - 输出状态价值
        self.critic = nn.Linear(512, 1)

    def forward(self, x):
        x = x.float() / 255
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # 将卷积层的输出展平
        x = F.relu(self.fc(x))
        
        # 输出动作概率和状态价值
        action_probs = F.softmax(self.actor(x), dim=-1)
        state_value = self.critic(x)
        
        return action_probs, state_value

class A2CAgent:
    def __init__(self, in_channels=4, action_space=[], learning_rate_actor=LEARNING_RATE_ACTOR, 
                 learning_rate_critic=LEARNING_RATE_CRITIC, trained_model_path=''):
        self.in_channels = in_channels
        self.action_space = action_space
        self.action_dim = self.action_space.n
        
        # 创建A2C网络
        self.network = A2CNetwork(self.in_channels, self.action_dim).to(device)
        
        # 使用两个不同的优化器
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate_actor)
        
        # 加载预训练模型（如果有）
        if trained_model_path != '':
            self.network.load_state_dict(torch.load(trained_model_path))
        
        # 保存一个episode的轨迹用于更新
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.entropies = []
        
        self.stepdone = 0
    
    def select_action(self, state):
        self.stepdone += 1
        state = state.to(device)
        
        # 在训练状态下需要梯度，仅在选择动作时不需要
        with torch.no_grad():
            action_probs, state_value = self.network(state)
        
        # 创建分布并采样
        dist = Categorical(action_probs)
        action = dist.sample()
        
        # 计算对数概率和熵
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, state_value, entropy
    
    def store_transition(self, state, action, reward, log_prob, value, entropy):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.entropies.append(entropy)
    
    def calculate_returns(self, next_value):
        # 计算每一步的回报
        returns = []
        R = next_value
        for reward in reversed(self.rewards):
            R = reward + GAMMA * R
            returns.insert(0, R)
        return returns
    
    def update_parameters(self, next_value):
        # 计算折扣回报
        returns = self.calculate_returns(next_value)
        returns = torch.cat(returns).detach()
        
        # 转换为tensor并确保requires_grad=True
        log_probs = torch.cat(self.log_probs)
        values = torch.cat(self.values)
        entropies = torch.cat(self.entropies)
        
        # 确保优势函数有梯度
        advantages = returns - values
        
        # 重新计算动作概率和值以获取梯度图
        states = torch.cat(self.states).to(device)
        actions = torch.cat(self.actions).to(device)
        
        # 前向传播以重建梯度图
        action_probs, recalculated_values = self.network(states)
        dist = Categorical(action_probs)
        recalculated_log_probs = dist.log_prob(actions)
        recalculated_entropy = dist.entropy().mean()
        
        # Actor loss: 使用重新计算的概率
        actor_loss = -(recalculated_log_probs * advantages.detach()).mean()
        
        # Critic loss: 使用重新计算的值，确保维度一致
        # squeeze确保两个张量都是一维的
        returns = returns.squeeze()
        critic_loss = F.mse_loss(recalculated_values.squeeze(), returns)
        
        # 总损失
        loss = actor_loss + VALUE_LOSS_COEF * critic_loss - ENTROPY_COEF * recalculated_entropy
        
        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), MAX_GRAD_NORM)
        self.optimizer.step()
        
        # 清空轨迹
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.entropies = []
        
        return actor_loss.item(), critic_loss.item()

class Trainer():
    def __init__(self, env, agent, n_episode):
        self.env = env
        self.n_episode = n_episode
        self.agent = agent
        self.rewardlist = []
        self.avg_rewardlist = []
        self.episode_rewards = []  # 用于计算最近100轮的平均奖励
        
        # 初始化图表数据
        self.episode_rewards = []
        self.avg_rewards = []
        self.episodes = []
        
        # 创建保存目录
        if not os.path.exists('training_visualization_a2c'):
            os.makedirs('training_visualization_a2c')
            os.makedirs('training_visualization_a2c/game_screens')
            os.makedirs('training_visualization_a2c/progress_plots')

    def save_visualization(self, frame, episode, reward, avg_reward):
        # 保存游戏画面
        if episode % 10 == 0:  # 每10轮保存一次游戏画面
            cv2.imwrite(f'training_visualization_a2c/game_screens/episode_{episode}.png', frame)
        
        # 更新训练进度数据
        self.episodes.append(episode)
        self.episode_rewards.append(reward)
        self.avg_rewards.append(avg_reward)
        
        # 每10轮保存一次训练进度图
        if episode % 10 == 0:
            plt.figure(figsize=(10, 5))
            
            # 确保数据是numpy数组
            episodes = np.array(self.episodes)
            rewards = np.array(self.episode_rewards)
            avg_rewards = np.array(self.avg_rewards)
            
            # 确保所有数据长度一致
            min_length = min(len(episodes), len(rewards), len(avg_rewards))
            episodes = episodes[:min_length]
            rewards = rewards[:min_length]
            avg_rewards = avg_rewards[:min_length]
            
            plt.clf()
            plt.plot(episodes, avg_rewards, 'b-', label='Average Reward')
            plt.plot(episodes, rewards, 'r-', label='Episode Reward', alpha=0.3)
            plt.xlabel('Episodes')
            plt.ylabel('Reward')
            plt.title(f'A2C Training Progress (Episode {episode})')
            plt.grid(True)
            plt.legend()
            plt.savefig(f'training_visualization_a2c/progress_plots/progress_{episode}.png')
            plt.close()

    def get_state(self, obs):
        state = np.array(obs)
        state = torch.from_numpy(state)
        return state.unsqueeze(0)

    def train(self):
        for episode in range(self.n_episode):
            obs = self.env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            state = np.stack((obs[0], obs[1], obs[2], obs[3]))
            state = self.get_state(state)
            episode_reward = 0.0
            
            for t in count():
                # 选择动作
                action, log_prob, value, entropy = self.agent.select_action(state)
                
                # 执行动作
                obs, reward, done, truncated, info = self.env.step(action.item())
                if isinstance(obs, tuple):
                    obs = obs[0]
                
                episode_reward += reward
                
                # 保存游戏画面和训练进度
                frame = obs[0]
                self.save_visualization(frame, episode, episode_reward, 
                                     sum(self.episode_rewards) / len(self.episode_rewards) if self.episode_rewards else 0)
                
                # 存储轨迹
                self.agent.store_transition(state, action, reward, log_prob, value, entropy)
                
                if not done:
                    next_state = np.stack((obs[0], obs[1], obs[2], obs[3]))
                    next_state = self.get_state(next_state)
                    state = next_state
                else:
                    break
            
            # 计算最后状态的价值（如果游戏结束，价值为0）
            if done:
                next_value = torch.zeros(1, 1, device=device)
            else:
                with torch.no_grad():
                    _, next_value = self.agent.network(state)
            
            # 更新网络参数
            actor_loss, critic_loss = self.agent.update_parameters(next_value)
            
            # 计算最近100轮的平均奖励
            self.episode_rewards.append(episode_reward)
            if len(self.episode_rewards) > 100:
                self.episode_rewards.pop(0)
            avg_reward = sum(self.episode_rewards) / len(self.episode_rewards)
            
            # 每100轮输出一次
            if episode % 100 == 0:
                print(f'Episode: {episode}/{self.n_episode} | 平均奖励: {avg_reward:.2f} | 当前奖励: {episode_reward:.2f} | actor_loss: {actor_loss:.4f} | critic_loss: {critic_loss:.4f}')
            
            if episode % 20 == 0:
                torch.save(self.agent.network.state_dict(), MODEL_STORE_PATH + '/' + "{}_episode{}.pt".format(modelname, episode))
            
            self.rewardlist.append(episode_reward)
            self.avg_rewardlist.append(episode_reward/t if t > 0 else 0)  # 避免除零错误
            
            self.env.close()
        return

    # 绘制单幕总奖励曲线
    def plot_total_reward(self):
        plt.plot(self.rewardlist)
        plt.xlabel("Training epochs")
        plt.ylabel("Total reward per episode")
        plt.title('Total reward curve of A2C on Breakout')
        plt.savefig('A2C_train_total_reward.png')
        plt.show()

    # 绘制单幕平均奖励曲线
    def plot_avg_reward(self):
        plt.plot(self.avg_rewardlist)
        plt.xlabel("Training epochs")
        plt.ylabel("Average reward per episode")
        plt.title('Average reward curve of A2C on Breakout')
        plt.savefig('A2C_train_avg_reward.png')
        plt.show()

# 从DQN复制环境包装器代码
class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)
    
class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = width
        self.height = height
        self.grayscale = grayscale
        if self.grayscale:
            self.observation_space = gym.spaces.Box(low=0, high=255,
                shape=(self.height, self.width, 1), dtype=np.uint8)
        else:
            self.observation_space = gym.spaces.Box(low=0, high=255,
                shape=(self.height, self.width, 3), dtype=np.uint8)

    def observation(self, frame):
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        if self.grayscale:
            frame = np.expand_dims(frame, -1)
        return frame
    
class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        return np.array(observation).astype(np.float32) / 255.0
    
class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(shp[:-1] + (shp[-1] * k,)), dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        if isinstance(ob, tuple):
            ob = ob[0]
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, truncated, info = self.env.step(action)
        if isinstance(ob, tuple):
            ob = ob[0]
        self.frames.append(ob)
        return self._get_ob(), reward, done, truncated, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))
    
class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.was_real_done = done
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            done = True
        self.lives = lives
        return obs, reward, done, truncated, info

    def reset(self, **kwargs):
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            obs, _, _, _, info = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info
    
class LazyFrames(object):
    def __init__(self, frames):
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[..., i]
    
def env_wrap_deepmind(env, episode_life=True, clip_rewards=True, frame_stack=True, scale=True):
    if episode_life:
        env = EpisodicLifeEnv(env)
    env = WarpFrame(env)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 4)
    return env

# 创建环境
env = env_wrap_deepmind(gym.make("ALE/Breakout-v5"), episode_life=True, clip_rewards=False, frame_stack=True, scale=False)
print(env)
action_space = env.action_space

# 创建A2C智能体
agent = A2CAgent(in_channels=4, action_space=action_space)

# 创建训练器并开始训练
trainer = Trainer(env, agent, n_episode)
trainer.train()
trainer.plot_total_reward()

# 保存奖励数据
np.save('total_reward_list_a2c_breakout.npy', np.array(trainer.rewardlist))
np.save('avg_reward_list_a2c_breakout.npy', np.array(trainer.avg_rewardlist))

print('The training costs {} episodes'.format(len(trainer.rewardlist)))
print('The max episode reward is {}, at episode {}'.format(
    max(trainer.rewardlist),
    trainer.rewardlist.index(max(trainer.rewardlist))
))

# 处理结果统计
if len(trainer.rewardlist) >= 5:
    # 合并5 episodes为1episode
    remainder = len(trainer.rewardlist) % 5
    if remainder > 0:
        trainer.rewardlist = trainer.rewardlist[:-remainder]  # 去除余数部分
    
    reshaped_reward_array = np.array(trainer.rewardlist).reshape((int(len(trainer.rewardlist)/5), 5))
    # 沿着第二个维度求和
    summed_reward_array = reshaped_reward_array.sum(axis=1)

    print('Now takes 5 episodes as 1, the training cost {} complete episodes'.format(len(summed_reward_array)))
    print('The max episode return is {}, at episode {}'.format(
        max(summed_reward_array),
        np.where(summed_reward_array == max(summed_reward_array))
    ))

    # 如果有足够多的样本，合并200 episodes为1epoch
    if len(summed_reward_array) >= 200:
        remainder = len(summed_reward_array) % 200
        if remainder > 0:
            summed_reward_array = summed_reward_array[:-remainder]  # 去除余数部分
            
        reshaped_reward_array_200 = summed_reward_array.reshape((int(len(summed_reward_array)/200), 200))
        # 沿着第二个维度求和
        summed_reward_array_200 = reshaped_reward_array_200.sum(axis=1)
        avg_reward_array_200 = summed_reward_array_200/200.0

        np.save('avg_reward_array_200_a2c_breakout.npy', avg_reward_array_200)

        print('The following graph takes 1000 games as 1 epoch where 5 games equals to 1 episode as stated before')

        max_idx = np.argmax(avg_reward_array_200)
        max_y = max(avg_reward_array_200)
        print('The best average return per epoch is {}, at epoch {}'.format(max_y, max_idx))

        plt.figure(figsize=(10,6))
        plt.plot(avg_reward_array_200, marker='o', markersize=4)
        plt.xlabel("Training Epochs", fontsize=12)
        plt.ylabel("Average Reward per Episode", fontsize=12)

        plt.scatter(max_idx, max_y, color='red', s=60)
        plt.annotate(f'max avg return: ({max_idx}, {max_y:.2f})', xy=(max_idx, max_y), xytext=(max_idx-1, max_y-1),
                    arrowprops=dict(facecolor='red', shrink=0.05))
        plt.savefig('A2C_train_total_reward_Breakout.svg')
        plt.show()
