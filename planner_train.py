#!/usr/bin/env python3
"""
Reinforcement Learning Training Script for Navigation Planner
Implements PPO algorithm for path planning optimization
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import yaml
import argparse
import logging
from pathlib import Path
import sys
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.environment import PathPlanningEnvironment, SimplePathPlanningEnv

# Add parent directory to path to import environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO)
module_name = os.path.splitext(os.path.basename(__file__))[0]
logger = logging.getLogger(module_name)


class ActorCritic(nn.Module):
    """
    Actor-Critic网络，用于同时学习策略(Actor)和价值函数(Critic)
    
    Args:
        state_dim (int): 状态空间维度
        action_dim (int): 动作空间维度
        hidden_dim (int): 隐藏层维度，默认为64
        num_layers (int): 网络层数，默认为2
        dropout_rate (float): Dropout比率，默认为0.0(不使用dropout)
    """
    def __init__(self, state_dim, action_dim, hidden_dim=64, num_layers=2, dropout_rate=0.0):
        super(ActorCritic, self).__init__()
        
        # 构建具有可配置深度的共享层
        shared_layers = []
        shared_layers.append(nn.Linear(state_dim, hidden_dim))
        shared_layers.append(nn.ReLU())
        
        for _ in range(num_layers - 1):
            shared_layers.append(nn.Linear(hidden_dim, hidden_dim))
            if dropout_rate > 0:
                shared_layers.append(nn.Dropout(dropout_rate))
            shared_layers.append(nn.ReLU())
        
        self.shared_layers = nn.Sequential(*shared_layers)
        
        # Actor头(策略) - 输出logits，而不是概率
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
            # 这里没有softmax - 按规范输出原始logits
        )
        
        # Critic头(价值函数)
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state, use_softmax=True):
        """
        前向传播函数
        
        Args:
            state (torch.Tensor): 输入状态
            use_softmax (bool): 是否返回softmax概率而不是logits
            
        Returns:
            tuple: (actions, value) 分别是动作分布(根据use_softmax参数返回logits或概率)和状态价值
        """
        shared_features = self.shared_layers(state)
        logits = self.actor_head(shared_features)  # 原始logits
        value = self.critic_head(shared_features)
        
        if use_softmax:
            # 在云训练中计算指标时使用softmax概率
            return torch.softmax(logits, dim=-1), value
        else:
            # 在训练过程中使用原始logits
            return logits, value  # 返回logits，而不是概率


class PPOAgent:
    """
    PPO(Proximal Policy Optimization)训练器
    
    Args:
        config (dict): 训练配置字典
        device (str): 训练设备，可选'cuda'或'cpu'
    """
    def __init__(self, config, device=None):
        self.config = config
        self.state_dim = config['state_dim']
        self.action_dim = config['action_dim']
        self.hidden_dim = config['hidden_dim']
        
        # 网络参数
        self.num_layers = config.get('network_layers', 2)
        self.dropout_rate = config.get('dropout_rate', 0.0)
        
        # 初始化网络
        self.actor_critic = ActorCritic(
            self.state_dim, 
            self.action_dim, 
            self.hidden_dim,
            self.num_layers,
            self.dropout_rate
        )
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=config['learning_rate'], eps=1e-5)
        
        # 学习率调度器
        self.use_lr_scheduler = config.get('use_lr_scheduler', False)
        if self.use_lr_scheduler:
            self.scheduler = optim.lr_scheduler.ExponentialLR(
                self.optimizer, 
                gamma=config.get('lr_decay_rate', 0.99)
            )
        
        # 超参数
        self.gamma = config['gamma']  # 折扣因子
        self.lam = config['lam']      # GAE lambda参数
        self.epsilon = config['epsilon']  # PPO裁剪参数
        self.epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.entropy_coef = config.get('entropy_coef', 0.01)
        
        # 梯度裁剪
        self.use_gradient_clipping = config.get('use_gradient_clipping', True)
        self.gradient_clip_value = config.get('gradient_clip_value', 0.5)
        
        # 设备配置 - 支持CPU和GPU训练，可根据用户选择决定
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.actor_critic = self.actor_critic.to(self.device)
        logger.info(f"Using device: {self.device}")

    def take_action(self, state):
        """
        根据当前策略采取动作
        
        Args:
            state (array-like): 当前状态
            
        Returns:
            int: 选取的动作
        """
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, _ = self.actor_critic(state_tensor)
        # Apply numerical stability measures
        logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
        logits = torch.clamp(logits, -10, 10)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item()
    
    def update(self, states, actions, rewards, old_log_probs, values, dones):
        """
        执行PPO更新
        
        Args:
            states (list): 状态序列
            actions (list): 动作序列
            rewards (list): 奖励序列
            old_log_probs (list): 旧的对数概率序列
            values (list): 状态价值序列
            dones (list): 结束标志序列
            
        Returns:
            tuple: (actor_loss, critic_loss) 分别是Actor和Critic的损失值
        """
        # 计算优势函数
        advantages = self.compute_gae(rewards, values, dones)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        
        # 标准化优势函数
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 转换为张量
        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32, device=self.device)
        values = torch.tensor(values, dtype=torch.float32, device=self.device)
        returns = advantages + values
        
        # PPO更新
        actor_losses = []
        critic_losses = []
        
        for _ in range(self.epochs):
            # 采样小批次
            indices = np.random.permutation(len(states))
            
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                # 获取批次数据
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # 前向传播
                logits, value = self.actor_critic(batch_states)
                # Apply numerical stability measures
                logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
                logits = torch.clamp(logits, -10, 10)
                    
                dist = Categorical(logits=logits)
                entropy = dist.entropy().mean()
                
                # 新的对数概率
                new_log_probs = dist.log_prob(batch_actions)
                
                # 比率
                ratio = torch.exp(torch.clamp(new_log_probs - batch_old_log_probs, -10, 10))
                
                # 代理损失
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Critic损失
                # Ensure both tensors have the same shape for MSE loss
                value_squeezed = value.squeeze()
                batch_returns_squeezed = batch_returns.squeeze() 
                critic_loss = nn.MSELoss()(value_squeezed, batch_returns_squeezed)
                
                # 总损失
                loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy
                
                # 更新
                self.optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪以保证稳定性
                if self.use_gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.gradient_clip_value)
                    
                self.optimizer.step()
                
                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                
        # 更新学习率
        if self.use_lr_scheduler:
            self.scheduler.step()
                
        avg_actor_loss = np.mean(actor_losses)
        avg_critic_loss = np.mean(critic_losses)
        
        logger.debug(f"PPO update completed. Actor Loss: {avg_actor_loss:.4f}, "
                     f"Critic Loss: {avg_critic_loss:.4f}")
        return avg_actor_loss, avg_critic_loss
    
    def compute_gae(self, rewards, values, dones):
        """
        计算广义优势估计(GAE)
        
        Args:
            rewards (list): 奖励序列
            values (list): 状态价值序列
            dones (list): 结束标志序列
            
        Returns:
            list: 优势函数值序列
        """
        advantages = []
        gae = 0
        
        # 为下一个状态添加虚拟值
        values = values + [0]
        
        for i in reversed(range(len(rewards))):
            if dones[i]:
                delta = rewards[i] - values[i]
                gae = delta
            else:
                delta = rewards[i] + self.gamma * values[i+1] - values[i]
                gae = delta + self.gamma * self.lam * gae
                
            advantages.insert(0, gae)
            
        return advantages


def load_training_config(config_path):
    """
    从YAML文件加载训练配置
    
    Args:
        config_path (str): 配置文件路径
        
    Returns:
        dict: 配置字典
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        logger.info(f"Using config: {config_path}")
    return config


def moving_average(a, window_size):
    """
    计算数组a的移动平均值
    
    Args:
        a (array-like): 输入数组
        window_size (int): 窗口大小
        
    Returns:
        numpy.ndarray: 移动平均后的数组
    """
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


def plot_training_results(return_list, env_name):
    """
    绘制训练结果
    
    Args:
        return_list (list): 回报序列
        env_name (str): 环境名称
    """
    
    episodes_list = list(range(len(return_list)))
    
    # 绘制原始回报
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('PPO on {}'.format(env_name))
    
    # 绘制平滑后的回报
    plt.subplot(1, 2, 2)
    if len(return_list) > 9:
        mv_return = moving_average(return_list, 9)
        plt.plot(episodes_list, mv_return)
    else:
        plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('PPO on {} (Smoothed)'.format(env_name))
    
    plt.tight_layout()
    plt.show()
    
def main():
    """
    主训练函数
    """
    parser = argparse.ArgumentParser(description='Train PPO agent for navigation planning')
    parser.add_argument('--config', type=str, default='config/ppo_config.yaml',
                        help='Path to training configuration file')
    parser.add_argument('--weights-path', type=str, default='models/ppo_weights.pt',
                        help='Path to save trained weights')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of training episodes (overrides config)')
    parser.add_argument('--env-type', type=str, default='simple',
                        choices=['simple', 'complex'],
                        help='Type of environment for training')
    parser.add_argument('--save-interval', type=int, default=1000,
                        help='Save model every N episodes')
    parser.add_argument('--plot-results', type=bool, default=True,
                        help='Plot training results after training')
    parser.add_argument('--device', type=str, default="cpu",
                        help='Device to use for training (cuda or cpu). If not specified, will use CUDA if available, otherwise CPU.')
    parser.add_argument('--data-sparse-factor', type=float, default=0.1,
                        help='Factor for data scarcity reward')
    parser.add_argument('--coverage-factor', type=float, default=0.1,
                        help='Factor for coverage reward')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_training_config(args.config)
    
    # 如果指定了episodes，则覆盖配置中的值
    if args.episodes is not None:
        config['episodes'] = args.episodes
    
    # 如果models目录不存在则创建
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # 初始化环境 - 现在支持新的参数
    if args.env_type == 'simple':
        env = SimplePathPlanningEnv(
            width=config.get('env_width', 50),
            height=config.get('env_height', 50),
            data_sparse_factor=args.data_sparse_factor,
            coverage_factor=args.coverage_factor
        )
    elif args.env_type == 'complex':
        env = PathPlanningEnvironment(
            width=config.get('env_width', 50),
            height=config.get('env_height', 50),
            obstacle_ratio=config.get('obstacle_ratio', 0.2),
            data_sparse_factor=args.data_sparse_factor,
            coverage_factor=args.coverage_factor
        )
    
    # 初始化训练器
    trainer = PPOAgent(config, args.device)
    
    logger.info("Starting PPO training...")
    logger.info(f"Training for {config['episodes']} episodes")
    logger.info(f"Using env-type: {args.env_type}")
    logger.info(f"Using data-sparse-factor: {args.data_sparse_factor}")
    logger.info(f"Using coverage-factor: {args.coverage_factor}")
    
    # 训练指标
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    # Track additional metrics for the new reward components
    data_scarcity_rewards = []
    coverage_rewards = []
    path_efficiency_penalties = []
    repeated_path_penalties = []
    collision_penalties = []
    
    # 定义每个进度条的剧集数(每200个剧集一个进度条)
    episodes_per_bar = 200
    total_episodes = config['episodes']
    
    # 带分段进度条的训练循环
    for episode in range(total_episodes):
        if episode % episodes_per_bar == 0:
            if 'pbar' in locals():
                pbar.close()

            remaining_episodes = min(episodes_per_bar, total_episodes - episode)
            pbar = tqdm(total=remaining_episodes, desc=f'Training Episodes {episode+1}-{min(episode+episodes_per_bar, total_episodes)}')
        
        # 收集轨迹数据
        states = []
        actions = []
        rewards = []
        log_probs = []
        values = []
        dones = []
        
        # 重置环境
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        
        # Track component rewards for this episode
        episode_data_scarcity_rewards = []
        episode_coverage_rewards = []
        episode_path_efficiency_penalties = []
        episode_repeated_path_penalties = []
        episode_collision_penalties = []
        
        # 收集剧集轨迹
        for step in range(config.get('max_steps', 200)):
            # 转换为张量
            state_tensor = torch.tensor(state, dtype=torch.float32, device=trainer.device).unsqueeze(0)
            
            # 获取动作概率和价值
            logits, value = trainer.actor_critic(state_tensor)
            # Apply numerical stability measures
            logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
            logits = torch.clamp(logits, -10, 10)
            
            # 直接使用logits构建Categorical分布
            dist = Categorical(logits=logits)
            
            # 采样动作
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            # 存储数据
            states.append(state.copy())
            actions.append(action.item())
            log_probs.append(log_prob.item())
            values.append(value.item())
            
            # 在环境中执行动作
            next_state, reward, done, info = env.step(action.item())
            state = next_state
            rewards.append(reward)
            dones.append(done)
            
            episode_reward += reward
            episode_length += 1
            
            # Track component rewards
            episode_data_scarcity_rewards.append(info.get('data_scarcity_reward', 0))
            episode_coverage_rewards.append(info.get('coverage_reward', 0))
            episode_path_efficiency_penalties.append(info.get('path_efficiency_penalty', 0))
            episode_repeated_path_penalties.append(info.get('repeated_path_penalty', 0))
            episode_collision_penalties.append(info.get('collision_penalty', 0))
            
            if done:
                success_count += 1
                break
        
        # 更新agent
        if len(states) > 0:
            actor_loss, critic_loss = trainer.update(states, actions, rewards, log_probs, values, dones)
        
        # 记录剧集指标
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Record component rewards for analysis
        data_scarcity_rewards.append(np.sum(episode_data_scarcity_rewards))
        coverage_rewards.append(np.sum(episode_coverage_rewards))
        path_efficiency_penalties.append(np.sum(episode_path_efficiency_penalties))
        repeated_path_penalties.append(np.sum(episode_repeated_path_penalties))
        collision_penalties.append(np.sum(episode_collision_penalties))
        
        # 使用当前指标更新进度条
        avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
        avg_length = np.mean(episode_lengths[-100:]) if len(episode_lengths) >= 100 else np.mean(episode_lengths)
        success_rate = success_count / max(episode, 1)  # 避免除零错误
        
        # 计算平均奖励组件
        avg_data_scarcity = np.mean(data_scarcity_rewards[-100:]) if len(data_scarcity_rewards) >= 100 else np.mean(data_scarcity_rewards)
        avg_coverage = np.mean(coverage_rewards[-100:]) if len(coverage_rewards) >= 100 else np.mean(coverage_rewards)
        
        # 使用ANSI转义码进行指标着色
        reward_str = f'{avg_reward:.2f}'
        length_str = f'{avg_length:.2f}'
        success_str = f'{success_rate:.2%}'
        data_scarcity_str = f'{avg_data_scarcity:.2f}'
        coverage_str = f'{avg_coverage:.2f}'
        
        # 根据性能应用颜色
        if avg_reward > 0:
            reward_colored = f'\033[92m{reward_str}\033[0m'  # 正奖励显示绿色
        else:
            reward_colored = f'\033[91m{reward_str}\033[0m'  # 负奖励显示红色
            
        # 长度显示黄色(中性颜色)
        length_colored = f'\033[93m{length_str}\033[0m'
        
        # 高成功率显示蓝色，低成功率显示红色
        if success_rate > 0.7:
            success_colored = f'\033[94m{success_str}\033[0m'  # 高成功率显示蓝色
        elif success_rate > 0.3:
            success_colored = f'\033[93m{success_str}\033[0m'  # 中等成功率显示黄色
        else:
            success_colored = f'\033[91m{success_str}\033[0m'  # 低成功率显示红色
        
        pbar.set_postfix_str(f'Avg Reward: {reward_colored} | Avg Length: {length_colored} | Success: {success_colored} | Data: {data_scarcity_str} | Cov: {coverage_str}')
        
        # 每100个剧集标准日志记录
        # if episode % config.get('log_interval', 100) == 0:
        #     logger.info(f"Episode {episode}/{total_episodes}, "
        #                f"Avg Reward: {avg_reward:.2f}, "
        #                f"Avg Length: {avg_length:.2f}, "
        #                f"Success Rate: {success_rate:.2%}")
        
        # 定期保存模型
        # if episode > 0 and episode % args.save_interval == 0:
        #     checkpoint_path = models_dir / f"ppo_weights_ep_{episode}.pt"
        #     torch.save(trainer.actor_critic.state_dict(), checkpoint_path)
        #     logger.info(f"Checkpoint saved to {checkpoint_path}")
        
        # 更新进度条
        pbar.update(1)
    
    # 关闭最后一个进度条
    if 'pbar' in locals():
        pbar.close()
    
    # 保存最终权重
    torch.save(trainer.actor_critic.state_dict(), args.weights_path)
    logger.info(f"Model weights saved to {args.weights_path}")
    
    # 保存训练指标
    metrics = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'success_count': success_count,
        'data_scarcity_rewards': data_scarcity_rewards,
        'coverage_rewards': coverage_rewards,
        'path_efficiency_penalties': path_efficiency_penalties,
        'repeated_path_penalties': repeated_path_penalties,
        'collision_penalties': collision_penalties
    }
    
    metrics_path = models_dir / "training_metrics.npy"
    np.save(metrics_path, metrics)
    logger.info(f"Training metrics saved to {metrics_path}")

    ##绘制训练结果
    if args.plot_results:
        plot_training_results(episode_rewards, args.env_type)


if __name__ == "__main__":
    main()