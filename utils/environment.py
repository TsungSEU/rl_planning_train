#!/usr/bin/env python3
"""
Environment for path planning reinforcement learning
Provides a realistic 2D grid world with obstacles for navigation tasks
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Tuple, Optional


class PathPlanningEnvironment:
    """
    A 2D grid world environment for path planning tasks.
    The agent navigates from start to goal positions while avoiding obstacles.
    """
    
    def __init__(self, width: int = 20, height: int = 20, obstacle_ratio: float = 0.2):
        """
        Initialize the environment.
        
        Args:
            width: Width of the grid world
            height: Height of the grid world
            obstacle_ratio: Ratio of grid cells that are obstacles
        """
        self.width = width
        self.height = height
        self.obstacle_ratio = obstacle_ratio
        self.max_steps = 200  # Maximum steps per episode
        
        # Action space: 0=Right, 1=Up, 2=Left, 3=Down
        self.action_space = 4
        self.actions = {
            0: (1, 0),   # Right
            1: (0, 1),   # Up
            2: (-1, 0),  # Left
            3: (0, -1)   # Down
        }
        
        # State space: Extended state representation
        # [norm_lat, norm_lon, heatmap_summary(16), last_n_actions(4), 
        #  remaining_budget_norm, local_traffic_density]
        self.observation_space = 24  # Updated to match specification
        
        # Initialize map with obstacles
        self.reset_map()
        
        # Agent and goal positions
        self.agent_pos = None
        self.goal_pos = None
        self.step_count = 0
        self.trajectory = []  # Track agent's path
        self.action_history = []  # Track recent actions
        
    def reset_map(self):
        """Generate a new map with obstacles."""
        # Create obstacle map
        self.map = np.zeros((self.height, self.width), dtype=bool)
        
        # Randomly place obstacles
        num_obstacles = int(self.width * self.height * self.obstacle_ratio)
        obstacle_indices = np.random.choice(
            self.width * self.height, 
            num_obstacles, 
            replace=False
        )
        
        for idx in obstacle_indices:
            x = idx % self.width
            y = idx // self.width
            # Keep borders clear
            if 1 <= x < self.width - 1 and 1 <= y < self.height - 1:
                self.map[y, x] = True
    
    def reset(self, start_pos: Optional[Tuple[int, int]] = None, 
              goal_pos: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Reset the environment to initial state.
        
        Args:
            start_pos: Optional starting position (x, y)
            goal_pos: Optional goal position (x, y)
            
        Returns:
            Initial state (extended feature vector)
        """
        # Clear existing positions
        self.agent_pos = None
        self.goal_pos = None
        self.step_count = 0
        self.trajectory = []
        self.action_history = [0] * 4  # Initialize with no-op actions
        
        if start_pos is None:
            # Find a free cell for start position
            while self.agent_pos is None:
                x, y = np.random.randint(0, self.width), np.random.randint(0, self.height)
                if not self.map[y, x]:
                    self.agent_pos = np.array([x, y], dtype=float)
        else:
            self.agent_pos = np.array(start_pos, dtype=float)
            
        if goal_pos is None:
            # Find a free cell for goal position
            while self.goal_pos is None:
                x, y = np.random.randint(0, self.width), np.random.randint(0, self.height)
                if not self.map[y, x] and not np.array_equal([x, y], self.agent_pos):
                    self.goal_pos = np.array([x, y], dtype=float)
        else:
            self.goal_pos = np.array(goal_pos, dtype=float)
            
        self.trajectory.append(self.agent_pos.copy())
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """
        Construct the full state representation according to specification.
        
        Returns:
            State vector with 24 dimensions:
            [norm_lat, norm_lon, heatmap_summary(16), last_n_actions(4), 
             remaining_budget_norm, local_traffic_density]
        """
        # 1. Normalized coordinates (indices 0-1)
        norm_lat = self.agent_pos[0] / (self.width - 1)  # Normalize to [0,1]
        norm_lon = self.agent_pos[1] / (self.height - 1)  # Normalize to [0,1]
        
        # 2. Heatmap summary (indices 2-17, 16 values)
        # Simulate a local heatmap based on distance to goal and obstacles
        heatmap = self._generate_local_heatmap()
        
        # 3. Last N actions (indices 18-21, 4 values)
        # One-hot encoding of last 4 actions
        last_actions = np.zeros(4)
        for i, action in enumerate(self.action_history[-4:]):
            if 0 <= action < 4:  # Valid action
                last_actions[action] = 1
        
        # 4. Remaining budget (index 22)
        remaining_budget_norm = max(0, 1.0 - (self.step_count / self.max_steps))
        
        # 5. Local traffic density (index 23)
        local_traffic_density = self._calculate_local_density()
        
        # Combine all features
        state = np.concatenate([
            [norm_lat, norm_lon],          # Indices 0-1
            heatmap,                       # Indices 2-17
            last_actions,                  # Indices 18-21
            [remaining_budget_norm],       # Index 22
            [local_traffic_density]        # Index 23
        ])
        
        return state
    
    def _generate_local_heatmap(self, radius=3) -> np.ndarray:
        """
        Generate a simulated heatmap around the agent position.
        
        Args:
            radius: Radius of the local area to consider
            
        Returns:
            16-element array representing heatmap summary
        """
        # Create a local grid around the agent
        heatmap = np.zeros((2*radius+1, 2*radius+1))
        
        agent_x, agent_y = int(self.agent_pos[0]), int(self.agent_pos[1])
        
        for dy in range(-radius, radius+1):
            for dx in range(-radius, radius+1):
                x, y = agent_x + dx, agent_y + dy
                
                # Check bounds
                if 0 <= x < self.width and 0 <= y < self.height:
                    # Distance-based value
                    distance_to_agent = np.sqrt(dx**2 + dy**2)
                    value = 1.0 / (1.0 + distance_to_agent)
                    
                    # Reduce value if obstacle
                    if self.map[y, x]:
                        value *= 0.1
                    
                    # Increase value if goal is nearby
                    goal_distance = np.sqrt((x - self.goal_pos[0])**2 + (y - self.goal_pos[1])**2)
                    if goal_distance < radius:
                        value *= (2.0 - goal_distance/radius)
                        
                    heatmap[dy + radius, dx + radius] = value
        
        # Flatten and take every 2nd element to get 16 values from 49
        flattened = heatmap.flatten()
        # Take 16 evenly spaced elements
        indices = np.linspace(0, len(flattened)-1, 16, dtype=int)
        heatmap_summary = flattened[indices]
        
        # Normalize
        if np.max(heatmap_summary) > 0:
            heatmap_summary = heatmap_summary / np.max(heatmap_summary)
            
        return heatmap_summary
    
    def _calculate_local_density(self, radius=2) -> float:
        """
        Calculate local traffic density based on obstacles in vicinity.
        
        Args:
            radius: Radius to check for obstacles
            
        Returns:
            Density value normalized to [0,1]
        """
        agent_x, agent_y = int(self.agent_pos[0]), int(self.agent_pos[1])
        obstacle_count = 0
        total_cells = 0
        
        for dy in range(-radius, radius+1):
            for dx in range(-radius, radius+1):
                x, y = agent_x + dx, agent_y + dy
                total_cells += 1
                
                # Check bounds and obstacles
                if 0 <= x < self.width and 0 <= y < self.height and self.map[y, x]:
                    obstacle_count += 1
                    
        return obstacle_count / max(1, total_cells)  # Normalize to [0,1]
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute an action in the environment.
        
        Args:
            action: Action to execute
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        self.step_count += 1
        
        # Store action history
        self.action_history.append(action)
        if len(self.action_history) > 10:  # Keep only recent actions
            self.action_history = self.action_history[-10:]
        
        # Calculate new position
        dx, dy = self.actions[action]
        new_pos = self.agent_pos + [dx, dy]
        
        # Check boundaries
        new_x, new_y = new_pos
        if (0 <= new_x < self.width and 0 <= new_y < self.height and 
            not self.map[int(new_y), int(new_x)]):
            # Valid move
            old_pos = self.agent_pos.copy()
            self.agent_pos = new_pos
        else:
            # Invalid move (hit wall or obstacle)
            old_pos = self.agent_pos.copy()
            
        self.trajectory.append(self.agent_pos.copy())
        
        # Calculate reward
        old_distance = np.linalg.norm(old_pos - self.goal_pos)
        new_distance = np.linalg.norm(self.agent_pos - self.goal_pos)
        
        # Reward for getting closer to goal
        distance_reward = (old_distance - new_distance) * 2.0
        
        # Small time penalty to encourage efficiency
        time_penalty = -0.1
        
        # Reward for reaching goal
        goal_reward = 0.0
        done = False
        if new_distance < 1.0:
            goal_reward = 10.0
            done = True
            
        # Penalty for hitting walls or obstacles
        collision_penalty = 0.0
        if np.array_equal(old_pos, self.agent_pos):
            collision_penalty = -1.0
            
        reward = distance_reward + time_penalty + goal_reward + collision_penalty
        
        # Check if max steps reached
        if self.step_count >= self.max_steps:
            done = True
            
        info = {
            'distance_to_goal': new_distance,
            'distance_reward': distance_reward,
            'goal_reward': goal_reward,
            'collision_penalty': collision_penalty
        }
        
        return self._get_state(), reward, done, info
    
    def render(self, ax=None, save_path: Optional[str] = None):
        """
        Render the environment.
        
        Args:
            ax: Matplotlib axis to render on (optional)
            save_path: Path to save figure (optional)
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            
        # Draw obstacles
        for y in range(self.height):
            for x in range(self.width):
                if self.map[y, x]:
                    rect = patches.Rectangle((x-0.5, y-0.5), 1, 1, 
                                           linewidth=0, facecolor='black', alpha=0.8)
                    ax.add_patch(rect)
                    
        # Draw agent
        agent_circle = plt.Circle(self.agent_pos, 0.3, color='blue')
        ax.add_patch(agent_circle)
        
        # Draw goal
        goal_square = patches.Rectangle((self.goal_pos[0]-0.5, self.goal_pos[1]-0.5), 
                                       1, 1, linewidth=2, edgecolor='green', facecolor='none')
        ax.add_patch(goal_square)
        
        # Formatting
        ax.set_xlim(-0.5, self.width-0.5)
        ax.set_ylim(-0.5, self.height-0.5)
        ax.set_aspect('equal')
        ax.grid(True, color='gray', linestyle='-', linewidth=0.5)
        ax.set_xticks(range(self.width))
        ax.set_yticks(range(self.height))
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
        if ax is None:
            plt.show()


# Simple test environment for quick experiments
class SimplePathPlanningEnv:
    """
    Simplified environment without obstacles for basic testing.
    Updated to support extended state representation.
    """
    
    def __init__(self, width: int = 20, height: int = 20):
        self.width = width
        self.height = height
        self.max_steps = 200
        self.action_space = 4
        self.observation_space = 24  # Updated to match specification
        self.agent_pos = None
        self.goal_pos = None
        self.step_count = 0
        self.trajectory = []
        self.action_history = []
        
        # Actions: 0=Right, 1=Up, 2=Left, 3=Down
        self.actions = {
            0: (1, 0),
            1: (0, 1),
            2: (-1, 0),
            3: (0, -1)
        }
        
    def reset(self, start_pos=None, goal_pos=None):
        self.step_count = 0
        self.trajectory = []
        self.action_history = [0] * 4
        
        if start_pos is None:
            self.agent_pos = np.array([0.0, 0.0])
        else:
            self.agent_pos = np.array(start_pos, dtype=float)
            
        if goal_pos is None:
            self.goal_pos = np.array([float(self.width-1), float(self.height-1)])
        else:
            self.goal_pos = np.array(goal_pos, dtype=float)
            
        self.trajectory.append(self.agent_pos.copy())
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """
        Construct the full state representation according to specification.
        
        Returns:
            State vector with 24 dimensions:
            [norm_lat, norm_lon, heatmap_summary(16), last_n_actions(4), 
             remaining_budget_norm, local_traffic_density]
        """
        # 1. Normalized coordinates (indices 0-1)
        norm_lat = self.agent_pos[0] / (self.width - 1)  # Normalize to [0,1]
        norm_lon = self.agent_pos[1] / (self.height - 1)  # Normalize to [0,1]
        
        # 2. Heatmap summary (indices 2-17, 16 values)
        # In simple environment, generate a basic heatmap based on distance to goal
        heatmap = self._generate_simple_heatmap()
        
        # 3. Last N actions (indices 18-21, 4 values)
        last_actions = np.zeros(4)
        for i, action in enumerate(self.action_history[-4:]):
            if 0 <= action < 4:  # Valid action
                last_actions[action] = 1
        
        # 4. Remaining budget (index 22)
        remaining_budget_norm = max(0, 1.0 - (self.step_count / self.max_steps))
        
        # 5. Local traffic density (index 23)
        # No obstacles in simple environment
        local_traffic_density = 0.0
        
        # Combine all features
        state = np.concatenate([
            [norm_lat, norm_lon],          # Indices 0-1
            heatmap,                       # Indices 2-17
            last_actions,                  # Indices 18-21
            [remaining_budget_norm],       # Index 22
            [local_traffic_density]        # Index 23
        ])
        
        return state
    
    def _generate_simple_heatmap(self, radius=3) -> np.ndarray:
        """
        Generate a simple heatmap for the basic environment.
        """
        # Create a local grid around the agent
        heatmap = np.zeros((2*radius+1, 2*radius+1))
        
        agent_x, agent_y = int(self.agent_pos[0]), int(self.agent_pos[1])
        
        for dy in range(-radius, radius+1):
            for dx in range(-radius, radius+1):
                x, y = agent_x + dx, agent_y + dy
                
                # Check bounds
                if 0 <= x < self.width and 0 <= y < self.height:
                    # Distance-based value to goal
                    distance_to_goal = np.sqrt((x - self.goal_pos[0])**2 + (y - self.goal_pos[1])**2)
                    value = 1.0 / (1.0 + distance_to_goal/5.0)
                    heatmap[dy + radius, dx + radius] = value
        
        # Flatten and take every 2nd element to get 16 values from 49
        flattened = heatmap.flatten()
        # Take 16 evenly spaced elements
        indices = np.linspace(0, len(flattened)-1, 16, dtype=int)
        heatmap_summary = flattened[indices]
        
        # Normalize
        if np.max(heatmap_summary) > 0:
            heatmap_summary = heatmap_summary / np.max(heatmap_summary)
            
        return heatmap_summary
    
    def step(self, action):
        self.step_count += 1
        
        # Store action history
        self.action_history.append(action)
        if len(self.action_history) > 10:  # Keep only recent actions
            self.action_history = self.action_history[-10:]
        
        # Store old position
        old_pos = self.agent_pos.copy()
        
        # Calculate new position
        dx, dy = self.actions[action]
        new_pos = self.agent_pos + [dx, dy]
        
        # Boundary checking
        new_pos[0] = np.clip(new_pos[0], 0, self.width - 1)
        new_pos[1] = np.clip(new_pos[1], 0, self.height - 1)
        
        self.agent_pos = new_pos
        self.trajectory.append(self.agent_pos.copy())
        
        # Calculate reward based on distance change
        old_distance = np.linalg.norm(old_pos - self.goal_pos)
        new_distance = np.linalg.norm(self.agent_pos - self.goal_pos)
        
        # Reward for getting closer to goal
        distance_reward = (old_distance - new_distance) * 2.0
        
        # Small time penalty to encourage efficiency
        time_penalty = -0.1
        
        # Reward for reaching goal
        goal_reward = 0.0
        done = False
        if new_distance < 0.5:
            goal_reward = 10.0
            done = True
            
        # Penalty for ineffective moves (hitting walls)
        wall_penalty = 0.0
        if np.array_equal(old_pos, self.agent_pos):
            wall_penalty = -1.0
            
        reward = distance_reward + time_penalty + goal_reward + wall_penalty
        
        # Check if max steps reached
        if self.step_count >= self.max_steps:
            done = True
            
        info = {
            'distance_to_goal': new_distance,
            'distance_reward': distance_reward,
            'goal_reward': goal_reward,
            'wall_penalty': wall_penalty
        }
        return self._get_state(), reward, done, info
    
    def render(self, ax=None, save_path: Optional[str] = None):
        """
        Render the simple environment.
        
        Args:
            ax: Matplotlib axis to render on (optional)
            save_path: Path to save figure (optional)
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            
        # Draw agent
        agent_circle = plt.Circle(self.agent_pos, 0.3, color='blue')
        ax.add_patch(agent_circle)
        
        # Draw goal
        goal_square = patches.Rectangle((self.goal_pos[0]-0.5, self.goal_pos[1]-0.5), 
                                       1, 1, linewidth=2, edgecolor='green', facecolor='none')
        ax.add_patch(goal_square)
        
        # Formatting
        ax.set_xlim(-0.5, self.width-0.5)
        ax.set_ylim(-0.5, self.height-0.5)
        ax.set_aspect('equal')
        ax.grid(True, color='gray', linestyle='-', linewidth=0.5)
        ax.set_xticks(range(self.width))
        ax.set_yticks(range(self.height))
        ax.set_title("Simple Environment (No Obstacles)")
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
        if ax is None:
            plt.show()