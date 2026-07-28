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
import matplotlib.pyplot as plt
import cv2
from itertools import count
import random, pickle, os.path, math, glob

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# PPO超参数
BATCH_SIZE = 128
GAMMA = 0.99
EPSILON = 0.2
LAMBDA = 0.95
LEARNING_RATE = 3e-4
UPDATE_EPOCHS = 10
CLIP_EPSILON = 0.2
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5
n_episode = 20000

MODEL_STORE_PATH = 'models'
modelname = 'PPO_Breakout'

class PPOActorCritic(nn.Module):
    def __init__(self, in_channels=4, n_actions=14):
        super(PPOActorCritic, self).__init__()
        # 特征提取网络
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(7*7*64, 512)
        
        # Actor网络 (策略网络)
        self.actor = nn.Linear(512, n_actions)
        
        # Critic网络 (价值网络)
        self.critic = nn.Linear(512, 1)
        
    def forward(self, x):
        x = x.float() / 255
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        
        # 输出动作概率和价值估计
        action_probs = F.softmax(self.actor(x), dim=-1)
        state_value = self.critic(x)
        
        return action_probs, state_value

class PPOAgent:
    def __init__(self, in_channels=4, action_space=[], learning_rate=LEARNING_RATE):
        self.in_channels = in_channels
        self.action_space = action_space
        self.action_dim = self.action_space.n
        
        self.actor_critic = PPOActorCritic(self.in_channels, self.action_dim).to(device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        
        # 经验缓冲区
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.masks = []
        
    def select_action(self, state):
        state = state.to(device)
        action_probs, state_value = self.actor_critic(state)
        
        # 创建动作分布
        dist = Categorical(action_probs)
        
        # 采样动作
        action = dist.sample()
        
        # 计算动作的log概率
        log_prob = dist.log_prob(action)
        
        return action, log_prob, state_value
    
    def store_transition(self, state, action, reward, value, log_prob, mask):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.masks.append(mask)
    
    def compute_gae(self, next_value):
        gae = 0
        returns = []
        
        for step in reversed(range(len(self.rewards))):
            if step == len(self.rewards) - 1:
                next_value = next_value
            else:
                next_value = self.values[step + 1]
            
            delta = self.rewards[step] + GAMMA * next_value * self.masks[step] - self.values[step]
            gae = delta + GAMMA * LAMBDA * self.masks[step] * gae
            returns.insert(0, gae + self.values[step])
            
        return returns
    
    def update(self):
        # 计算优势函数
        next_value = 0
        returns = self.compute_gae(next_value)
        
        # 转换为tensor
        returns = torch.tensor(returns).to(device)
        old_values = torch.cat(self.values).to(device)
        old_log_probs = torch.cat(self.log_probs).to(device)
        actions = torch.cat(self.actions).to(device)
        states = torch.cat(self.states).to(device)
        
        # 计算优势
        advantages = returns - old_values
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 在每个epoch中，我们都从头开始计算，避免共享计算图
        for _ in range(UPDATE_EPOCHS):
            # 完全重新计算所有值，确保每次迭代都使用新的计算图
            self.optimizer.zero_grad()
            
            # 向前传播
            action_probs, value_preds = self.actor_critic(states)
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # 计算比率
            ratio = torch.exp(new_log_probs - old_log_probs.detach())
            
            # 计算PPO目标
            surr1 = ratio * advantages.detach()
            surr2 = torch.clamp(ratio, 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON) * advantages.detach()
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # 计算价值损失
            value_loss = F.mse_loss(value_preds.squeeze(), returns.detach())
            
            # 计算总损失
            loss = actor_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy
            
            # 反向传播和优化
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), MAX_GRAD_NORM)
            self.optimizer.step()
        
        # 清空缓冲区
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.masks = []

class Trainer():
    def __init__(self, env, agent, n_episode):
        self.env = env
        self.n_episode = n_episode
        self.agent = agent
        self.rewardlist = []
        self.avg_rewardlist = []
        self.episode_rewards = []
        
        # 初始化图表数据
        self.episode_rewards = []
        self.avg_rewards = []
        self.episodes = []
        
        # 创建保存目录
        if not os.path.exists('training_visualization'):
            os.makedirs('training_visualization')
            os.makedirs('training_visualization/game_screens')
            os.makedirs('training_visualization/progress_plots')

    def save_visualization(self, frame, episode, reward, avg_reward):
        # 保存游戏画面
        if episode % 10 == 0:  # 每10轮保存一次游戏画面
            cv2.imwrite(f'training_visualization/game_screens/episode_{episode}.png', frame)
        
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
            plt.title(f'PPO Training Progress (Episode {episode})')
            plt.grid(True)
            plt.legend()
            plt.savefig(f'training_visualization/progress_plots/progress_{episode}.png')
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
                action, log_prob, value = self.agent.select_action(state)
                
                # 执行动作
                obs, reward, done, truncated, info = self.env.step(action.item())
                if isinstance(obs, tuple):
                    obs = obs[0]
                episode_reward += reward

                # 保存游戏画面和训练进度
                frame = obs[0]
                self.save_visualization(frame, episode, episode_reward, 
                                     sum(self.episode_rewards) / len(self.episode_rewards) if self.episode_rewards else 0)

                # 存储转换
                mask = 1.0 if not done else 0.0
                self.agent.store_transition(state, action, reward, value, log_prob, mask)

                if not done:
                    next_state = np.stack((obs[0], obs[1], obs[2], obs[3]))
                    next_state = self.get_state(next_state)
                    state = next_state
                else:
                    break

            # 更新策略
            self.agent.update()

            # 计算最近100轮的平均奖励
            self.episode_rewards.append(episode_reward)
            if len(self.episode_rewards) > 100:
                self.episode_rewards.pop(0)
            avg_reward = sum(self.episode_rewards) / len(self.episode_rewards)

            # 每100轮输出一次
            if episode % 100 == 0:
                print(f'Episode: {episode}/{self.n_episode} | 平均奖励: {avg_reward:.2f} | 当前奖励: {episode_reward:.2f}')
            
            if episode % 20 == 0:
                torch.save(self.agent.actor_critic.state_dict(), MODEL_STORE_PATH + '/' + "{}_episode{}.pt".format(modelname, episode))

            self.rewardlist.append(episode_reward)
            self.avg_rewardlist.append(episode_reward/t)

            self.env.close()
        return

    def plot_total_reward(self):
        plt.plot(self.rewardlist)
        plt.xlabel("Training epochs")
        plt.ylabel("Total reward per episode")
        plt.title('Total reward curve of PPO on Breakout')
        plt.savefig('PPO_train_total_reward.png')
        plt.show()

    def plot_avg_reward(self):
        plt.plot(self.avg_rewardlist)
        plt.xlabel("Training epochs")
        plt.ylabel("Average reward per episode")
        plt.title('Average reward curve of PPO on Breakout')
        plt.savefig('PPO_train_avg_reward.png')
        plt.show()

# 环境包装器
class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        return np.sign(reward)

class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True):
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

# 创建环境和训练
env = env_wrap_deepmind(gym.make("ALE/Breakout-v5"), episode_life=True, clip_rewards=False, frame_stack=True, scale=False)
print(env)
action_space = env.action_space

agent = PPOAgent(in_channels=4, action_space=action_space)
trainer = Trainer(env, agent, n_episode)
trainer.train()
trainer.plot_total_reward()

# 保存训练数据
np.save('total_reward_list_ppo_breakout_1e5.npy', np.array(trainer.rewardlist))
np.save('avg_reward_list_ppo_breakout_1e5.npy', np.array(trainer.avg_rewardlist))

print('The training costs {} episodes'.format(len(trainer.rewardlist)))
print('The max episode reward is {}, at episode {}'.format(
    max(trainer.rewardlist),
    trainer.rewardlist.index(max(trainer.rewardlist))
))
