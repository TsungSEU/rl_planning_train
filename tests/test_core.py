#!/usr/bin/env python3
"""
Unit tests for core module.
"""

import sys
import os
import unittest
import numpy as np
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import PPOConfig, load_config
from core.base_agent import ActorCriticNet, ContinuousPPOAgent
from core.base_environment import BaseEnvironment
from core.base_reward import BaseRewardCalculator, RewardState, RewardBreakdown
from core.factory import ScenarioFactory


# Concrete reward calculator for testing
class ConcreteRewardCalculator(BaseRewardCalculator):
    """Concrete implementation for testing."""
    def compute(self, state):
        return RewardBreakdown(
            total=0.0,
            distance=self.distance_reward(state),
            time_penalty=self.time_penalty(state),
            goal=self.goal_reward(state),
            navigation=0.0,
        )


class TestConfig(unittest.TestCase):
    """Test configuration classes."""

    def test_ppo_config_nav_data(self):
        """Test PPOConfig for nav_data scenario."""
        config = PPOConfig(
            state_dim=43,
            action_dim=3,
            action_space='continuous',
            hidden_dim=256,
            network_layers=3,
            hidden_dims=[256, 128, 64],
            activation='ELU',
            scenario='nav_data'
        )
        self.assertEqual(config.state_dim, 43)
        self.assertEqual(config.action_dim, 3)
        self.assertEqual(config.scenario, 'nav_data')

    def test_load_config_nav_data(self):
        """Test loading nav_data config from YAML."""
        config = load_config('config/nav_data_training.yaml')
        self.assertEqual(config.state_dim, 43)
        self.assertEqual(config.action_dim, 3)
        self.assertEqual(config.scenario, 'nav_data')
        self.assertEqual(config.hidden_dims, [256, 128, 64])
        self.assertEqual(config.activation, 'ELU')


class TestActorCriticNet(unittest.TestCase):
    """Test ActorCritic network."""

    def test_custom_hidden_dims(self):
        """Test forward pass with custom hidden_dims."""
        net = ActorCriticNet(
            state_dim=43,
            action_dim=3,
            hidden_dims=[256, 128, 64],
            activation='ELU',
            action_space='continuous'
        )

        state = torch.randn(1, 43)
        action_mean, value = net(state, use_softmax=False)

        self.assertEqual(action_mean.shape, (1, 3))
        self.assertEqual(value.shape, (1, 1))


class TestContinuousPPOAgent(unittest.TestCase):
    """Test continuous PPO agent."""

    def setUp(self):
        self.config = PPOConfig(
            state_dim=43,
            action_dim=3,
            action_space='continuous',
            hidden_dim=128,
            network_layers=2,
            batch_size=32,
            epochs=2
        )
        self.agent = ContinuousPPOAgent(self.config, device='cpu')

    def test_select_action(self):
        state = np.random.randn(43)
        action, log_prob, value = self.agent.select_action(state)

        self.assertIsInstance(action, np.ndarray)
        self.assertEqual(action.shape, (3,))
        self.assertIsInstance(log_prob, float)
        self.assertIsInstance(value, float)

    def test_select_action_deterministic(self):
        state = np.random.randn(43)
        action1, _, _ = self.agent.select_action(state, deterministic=True)
        action2, _, _ = self.agent.select_action(state, deterministic=True)
        np.testing.assert_array_almost_equal(action1, action2)

    def test_take_action(self):
        """Test take_action convenience method."""
        state = np.random.randn(43)
        action = self.agent.take_action(state)
        self.assertIsInstance(action, np.ndarray)
        self.assertEqual(action.shape, (3,))

    def test_get_action_mean_std(self):
        """Test get_action_mean_std method."""
        state = np.random.randn(43)
        mean, std = self.agent.get_action_mean_std(state)
        self.assertEqual(mean.shape, (3,))
        self.assertEqual(std.shape, (3,))
        self.assertTrue(np.all(std > 0))

    def test_evaluate_batch(self):
        """Test evaluate_batch method."""
        states = [np.random.randn(43) for _ in range(4)]
        mean, values = self.agent.evaluate_batch(states)
        self.assertEqual(mean.shape, (4, 3))
        self.assertEqual(values.shape, (4, 1))

    def test_compute_gae(self):
        rewards = [1.0, 2.0, 3.0, 4.0, 10.0]
        values = [0.5, 1.0, 1.5, 2.0, 2.5]
        dones = [False, False, False, False, True]
        advantages = self.agent.compute_gae(rewards, values, dones)
        self.assertEqual(len(advantages), len(rewards))
        self.assertIsInstance(advantages[0], float)

    def test_update(self):
        states = [np.random.randn(43) for _ in range(64)]
        actions = [np.random.randn(3) for _ in range(64)]
        rewards = [np.random.randn() for _ in range(64)]
        old_log_probs = [np.random.randn() for _ in range(64)]
        values = [np.random.randn() for _ in range(64)]
        dones = [False] * 63 + [True]

        actor_loss, critic_loss = self.agent.update(
            states, actions, rewards, old_log_probs, values, dones
        )
        self.assertIsInstance(actor_loss, float)
        self.assertIsInstance(critic_loss, float)


class TestScenarioFactory(unittest.TestCase):
    """Test scenario factory."""

    def test_create_nav_data_env(self):
        config = PPOConfig(state_dim=43, action_dim=3, scenario='nav_data',
                          grid_resolution=0.5, env_width=60, env_height=60)
        env = ScenarioFactory.create_environment(config)
        self.assertEqual(env.state_dim, 43)
        self.assertEqual(env.action_dim, 3)

    def test_create_agent(self):
        config = PPOConfig(state_dim=43, action_dim=3, scenario='nav_data')
        agent = ScenarioFactory.create_agent(config, torch.device('cpu'))
        self.assertIsInstance(agent, ContinuousPPOAgent)


class TestRewardCalculator(unittest.TestCase):
    """Test reward calculator."""

    def setUp(self):
        self.config = {'w_distance': 2.0, 'w_time_penalty': 0.1}
        self.reward_calc = ConcreteRewardCalculator(self.config)

    def test_distance_reward(self):
        state = RewardState(
            position=np.array([5.0, 5.0]),
            goal_position=np.array([10.0, 10.0]),
            previous_position=np.array([0.0, 0.0])
        )
        reward = self.reward_calc.distance_reward(state)
        self.assertIsInstance(reward, float)

    def test_time_penalty(self):
        state = RewardState(position=np.array([0.0, 0.0]))
        penalty = self.reward_calc.time_penalty(state)
        self.assertEqual(penalty, -0.1)

    def test_goal_reward(self):
        state = RewardState(
            position=np.array([10.0, 10.0]),
            goal_position=np.array([10.0, 10.0]),
            is_goal_reached=True
        )
        reward = self.reward_calc.goal_reward(state)
        self.assertEqual(reward, 10.0)


if __name__ == '__main__':
    unittest.main()
