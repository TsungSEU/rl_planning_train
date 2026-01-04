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
    
    def __init__(self, width: int = 20, height: int = 20, obstacle_ratio: float = 0.2, 
                 data_sparse_factor: float = 0.1, coverage_factor: float = 0.1):
        """
        Initialize the environment.
        
        Args:
            width: Width of the grid world
            height: Height of the grid world
            obstacle_ratio: Ratio of grid cells that are obstacles
            data_sparse_factor: Factor for data scarcity reward
            coverage_factor: Factor for coverage reward
        """
        self.width = width
        self.height = height
        self.obstacle_ratio = obstacle_ratio
        self.max_steps = 200  # Maximum steps per episode
        self.data_sparse_factor = data_sparse_factor
        self.coverage_factor = coverage_factor
        
        # Action space: 0=Go Forward, 1=Turn Left, 2=Turn Right, 3=U-turn
        # The agent's direction determines movement direction
        self.action_space = 4  # Forward, Turn Left, Turn Right, U-turn
        self.directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # North, East, South, West
        self.current_direction = 0  # Start facing North (index 0)
        
        # State space: Extended state representation
        # [norm_lat, norm_lon, drivable_road_network(16), last_n_actions(4), 
        #  remaining_budget_norm, local_traffic_density]
        self.observation_space = 24  # Updated to match specification
        
        # Initialize map with obstacles (0=drivable, 1=obstacle)
        self.reset_map()
        
        # Create drivable road network (1=drivable, 0=non-drivable)
        self.drivable_network = np.ones((self.height, self.width), dtype=np.float32)
        self.drivable_network[self.map == True] = 0  # Set obstacles as non-drivable
        
        # Agent and goal positions
        self.agent_pos = None
        self.goal_pos = None
        self.step_count = 0
        self.trajectory = []  # Track agent's path
        self.action_history = []  # Track recent actions
        
        # For coverage and data scarcity metrics
        self.visited_cells = np.zeros((self.height, self.width), dtype=bool)
        self.data_sparse_map = np.random.rand(self.height, self.width)  # Random data scarcity values
        self.repeated_path_penalty = 0.0  # Track penalty for repeated paths
        self.previous_positions = []  # Track recent positions to detect repetition
        
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
        self.current_direction = 0  # Start facing North
        self.visited_cells = np.zeros((self.height, self.width), dtype=bool)
        self.repeated_path_penalty = 0.0
        self.previous_positions = []
        
        # Reset map and drivable network
        self.reset_map()
        self.drivable_network = np.ones((self.height, self.width), dtype=np.float32)
        self.drivable_network[self.map == True] = 0  # Set obstacles as non-drivable
        
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
            [norm_lat, norm_lon, drivable_road_network(16), last_n_actions(4), 
             remaining_budget_norm, local_traffic_density]
        """
        # 1. Normalized coordinates (indices 0-1)
        norm_lat = self.agent_pos[0] / (self.width - 1)  # Normalize to [0,1]
        norm_lon = self.agent_pos[1] / (self.height - 1)  # Normalize to [0,1]
        
        # 2. Drivable road network (indices 2-17, 16 values)
        # Extract drivable network around agent position
        drivable_network = self._extract_drivable_network_around_agent()
        
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
            drivable_network,              # Indices 2-17
            last_actions,                  # Indices 18-21
            [remaining_budget_norm],       # Index 22
            [local_traffic_density]        # Index 23
        ])
        
        return state
    
    def _extract_drivable_network_around_agent(self, radius=3) -> np.ndarray:
        """
        Extract drivable road network around agent position.
        
        Args:
            radius: Radius of the local area to consider
            
        Returns:
            16-element array representing drivable road network
        """
        # Create a local grid around the agent
        local_network = np.zeros((2*radius+1, 2*radius+1))
        
        agent_x, agent_y = int(self.agent_pos[0]), int(self.agent_pos[1])
        
        for dy in range(-radius, radius+1):
            for dx in range(-radius, radius+1):
                x, y = agent_x + dx, agent_y + dy
                
                # Check bounds
                if 0 <= x < self.width and 0 <= y < self.height:
                    # Get drivable status
                    local_network[dy + radius, dx + radius] = self.drivable_network[y, x]
        
        # Flatten and take 16 evenly spaced elements from the local network
        flattened = local_network.flatten()
        # Take 16 evenly spaced elements
        indices = np.linspace(0, len(flattened)-1, 16, dtype=int)
        drivable_network_summary = flattened[indices]
        
        return drivable_network_summary
    
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
    
    def _calculate_data_scarcity_reward(self) -> float:
        """Calculate reward based on data scarcity of current position."""
        agent_x, agent_y = int(self.agent_pos[0]), int(self.agent_pos[1])
        # Ensure we're within bounds
        agent_x = np.clip(agent_x, 0, self.width-1)
        agent_y = np.clip(agent_y, 0, self.height-1)
        return self.data_sparse_map[agent_y, agent_x] * self.data_sparse_factor
    
    def _calculate_coverage_reward(self) -> float:
        """Calculate reward based on coverage of environment."""
        coverage_ratio = np.sum(self.visited_cells) / (self.width * self.height)
        return coverage_ratio * self.coverage_factor
    
    def _calculate_path_efficiency_penalty(self) -> float:
        """Calculate penalty for inefficient path (too long)."""
        # Penalty increases with path length relative to direct distance
        direct_distance = np.linalg.norm(np.array(self.goal_pos) - np.array(self.trajectory[0]))
        path_length = len(self.trajectory)
        if direct_distance > 0:
            efficiency_ratio = path_length / (direct_distance + 1)  # +1 to avoid division by zero
            return -0.01 * efficiency_ratio  # Small penalty for inefficient paths
        return 0.0
    
    def _calculate_repeated_path_penalty(self) -> float:
        """Calculate penalty for visiting same positions repeatedly."""
        agent_x, agent_y = int(self.agent_pos[0]), int(self.agent_pos[1])
        if self.visited_cells[agent_y, agent_x]:
            return -0.05  # Small penalty for revisiting
        return 0.0
    
    def _calculate_collision_penalty(self) -> float:
        """Calculate penalty for collision with obstacles."""
        # Check if agent is in an obstacle cell
        agent_x, agent_y = int(self.agent_pos[0]), int(self.agent_pos[1])
        if 0 <= agent_x < self.width and 0 <= agent_y < self.height:
            if self.map[agent_y, agent_x]:  # If agent is in obstacle
                return -5.0  # Large penalty
        return 0.0
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute an action in the environment.
        
        Args:
            action: Action to execute (0=Forward, 1=Turn Left, 2=Turn Right, 3=U-turn)
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        self.step_count += 1
        
        # Store action history
        self.action_history.append(action)
        if len(self.action_history) > 10:  # Keep only recent actions
            self.action_history = self.action_history[-10:]
        
        # Store old position for distance calculation
        old_pos = self.agent_pos.copy()
        
        # Process the action
        if action == 0:  # Go Forward
            # Move in current direction
            dx, dy = self.directions[self.current_direction]
            new_pos = self.agent_pos + [dx, dy]
            
            # Check boundaries and obstacles
            new_x, new_y = new_pos
            if (0 <= new_x < self.width and 0 <= new_y < self.height and 
                not self.map[int(new_y), int(new_x)]):
                # Valid move
                self.agent_pos = new_pos
                # Mark this cell as visited
                agent_x, agent_y = int(self.agent_pos[0]), int(self.agent_pos[1])
                if 0 <= agent_x < self.width and 0 <= agent_y < self.height:
                    self.visited_cells[agent_y, agent_x] = True
        elif action == 1:  # Turn Left
            # Change direction counter-clockwise
            self.current_direction = (self.current_direction - 1) % 4
        elif action == 2:  # Turn Right
            # Change direction clockwise
            self.current_direction = (self.current_direction + 1) % 4
        elif action == 3:  # U-turn
            # Turn 180 degrees
            self.current_direction = (self.current_direction + 2) % 4
        
        self.trajectory.append(self.agent_pos.copy())
        
        # Calculate reward components
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
            
        # Penalty for hitting walls or obstacles (if attempted to move forward but couldn't)
        collision_penalty = 0.0
        if np.array_equal(old_pos, self.agent_pos) and action == 0:  # Tried to move forward but couldn't
            collision_penalty = -1.0
        else:
            # Check if agent is in obstacle after movement
            agent_x, agent_y = int(self.agent_pos[0]), int(self.agent_pos[1])
            if 0 <= agent_x < self.width and 0 <= agent_y < self.height and self.map[agent_y, agent_x]:
                collision_penalty = -5.0  # Large penalty for being in obstacle
        
        # NEW: Data scarcity reward
        data_scarcity_reward = self._calculate_data_scarcity_reward()
        
        # NEW: Coverage reward
        coverage_reward = self._calculate_coverage_reward()
        
        # NEW: Path efficiency penalty
        path_efficiency_penalty = self._calculate_path_efficiency_penalty()
        
        # NEW: Repeated path penalty
        repeated_path_penalty = self._calculate_repeated_path_penalty()
        
        # Total reward - only using specified components
        reward = (distance_reward + time_penalty + goal_reward + collision_penalty + 
                 data_scarcity_reward + coverage_reward + path_efficiency_penalty + 
                 repeated_path_penalty)
        
        # Check if max steps reached
        if self.step_count >= self.max_steps:
            done = True
            
        info = {
            'distance_to_goal': new_distance,
            'distance_reward': distance_reward,
            'goal_reward': goal_reward,
            'collision_penalty': collision_penalty,
            'data_scarcity_reward': data_scarcity_reward,
            'coverage_reward': coverage_reward,
            'path_efficiency_penalty': path_efficiency_penalty,
            'repeated_path_penalty': repeated_path_penalty,
            'current_direction': self.current_direction
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
                    
        # Draw visited cells (coverage)
        for y in range(self.height):
            for x in range(self.width):
                if self.visited_cells[y, x]:
                    rect = patches.Rectangle((x-0.5, y-0.5), 1, 1, 
                                           linewidth=0, facecolor='lightblue', alpha=0.3)
                    ax.add_patch(rect)
                    
        # Draw agent with direction indicator
        agent_circle = plt.Circle(self.agent_pos, 0.3, color='blue')
        ax.add_patch(agent_circle)
        
        # Draw direction indicator
        dir_dx, dir_dy = self.directions[self.current_direction]
        plt.arrow(self.agent_pos[0], self.agent_pos[1], dir_dx*0.3, dir_dy*0.3, 
                 head_width=0.1, head_length=0.1, fc='red', ec='red')
        
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
    
    def __init__(self, width: int = 20, height: int = 20, 
                 data_sparse_factor: float = 0.1, coverage_factor: float = 0.1):
        self.width = width
        self.height = height
        self.max_steps = 200
        self.action_space = 4  # Forward, Turn Left, Turn Right, U-turn
        self.directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # North, East, South, West
        self.current_direction = 0  # Start facing North (index 0)
        self.observation_space = 24  # Updated to match specification
        self.agent_pos = None
        self.goal_pos = None
        self.step_count = 0
        self.trajectory = []
        self.action_history = []
        self.data_sparse_factor = data_sparse_factor
        self.coverage_factor = coverage_factor
        
        # Initialize drivable network (all cells are drivable in simple environment)
        self.drivable_network = np.ones((self.height, self.width), dtype=np.float32)
        
        # For coverage and data scarcity metrics
        self.visited_cells = np.zeros((self.height, self.width), dtype=bool)
        self.data_sparse_map = np.random.rand(self.height, self.width)  # Random data scarcity values
        self.repeated_path_penalty = 0.0  # Track penalty for repeated paths
        self.previous_positions = []  # Track recent positions to detect repetition
        
        # Actions: 0=Forward, 1=Turn Left, 2=Turn Right, 3=U-turn
        self.actions = {
            0: "Forward",   # Go forward in current direction
            1: "Turn Left", # Turn counter-clockwise
            2: "Turn Right", # Turn clockwise
            3: "U-turn"     # Turn 180 degrees
        }
        
    def reset(self, start_pos=None, goal_pos=None):
        self.step_count = 0
        self.trajectory = []
        self.action_history = [0] * 4
        self.current_direction = 0  # Start facing North
        self.visited_cells = np.zeros((self.height, self.width), dtype=bool)
        self.repeated_path_penalty = 0.0
        self.previous_positions = []
        
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
            [norm_lat, norm_lon, drivable_road_network(16), last_n_actions(4), 
             remaining_budget_norm, local_traffic_density]
        """
        # 1. Normalized coordinates (indices 0-1)
        norm_lat = self.agent_pos[0] / (self.width - 1)  # Normalize to [0,1]
        norm_lon = self.agent_pos[1] / (self.height - 1)  # Normalize to [0,1]
        
        # 2. Drivable road network (indices 2-17, 16 values)
        # Extract drivable network around agent position
        drivable_network = self._extract_drivable_network_around_agent()
        
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
            [norm_lat, norm_lon],          # Indices 0-1, 位置坐标
            drivable_network,              # Indices 2-17, 可行驶路网
            last_actions,                  # Indices 18-21, 历史动作
            [remaining_budget_norm],       # Index 22, 预算信息
            [local_traffic_density]        # Index 23, 局部密度
        ])
        
        return state
    
    def _extract_drivable_network_around_agent(self, radius=3) -> np.ndarray:
        """
        Extract drivable road network around agent position.
        
        Args:
            radius: Radius of the local area to consider
            
        Returns:
            16-element array representing drivable road network
        """
        # Create a local grid around the agent
        local_network = np.zeros((2*radius+1, 2*radius+1))
        
        agent_x, agent_y = int(self.agent_pos[0]), int(self.agent_pos[1])
        
        for dy in range(-radius, radius+1):
            for dx in range(-radius, radius+1):
                x, y = agent_x + dx, agent_y + dy
                
                # Check bounds
                if 0 <= x < self.width and 0 <= y < self.height:
                    # Get drivable status (all are drivable in simple environment)
                    local_network[dy + radius, dx + radius] = self.drivable_network[y, x]
        
        # Flatten and take 16 evenly spaced elements from the local network
        flattened = local_network.flatten()
        # Take 16 evenly spaced elements
        indices = np.linspace(0, len(flattened)-1, 16, dtype=int)
        drivable_network_summary = flattened[indices]
        
        return drivable_network_summary
    
    def _calculate_data_scarcity_reward(self) -> float:
        """Calculate reward based on data scarcity of current position."""
        agent_x, agent_y = int(self.agent_pos[0]), int(self.agent_pos[1])
        # Ensure we're within bounds
        agent_x = np.clip(agent_x, 0, self.width-1)
        agent_y = np.clip(agent_y, 0, self.height-1)
        return self.data_sparse_map[agent_y, agent_x] * self.data_sparse_factor
    
    def _calculate_coverage_reward(self) -> float:
        """Calculate reward based on coverage of environment."""
        coverage_ratio = np.sum(self.visited_cells) / (self.width * self.height)
        return coverage_ratio * self.coverage_factor
    
    def _calculate_path_efficiency_penalty(self) -> float:
        """Calculate penalty for inefficient path (too long)."""
        # Penalty increases with path length relative to direct distance
        direct_distance = np.linalg.norm(np.array(self.goal_pos) - np.array(self.trajectory[0]))
        path_length = len(self.trajectory)
        if direct_distance > 0:
            efficiency_ratio = path_length / (direct_distance + 1)  # +1 to avoid division by zero
            return -0.01 * efficiency_ratio  # Small penalty for inefficient paths
        return 0.0
    
    def _calculate_repeated_path_penalty(self) -> float:
        """Calculate penalty for visiting same positions repeatedly."""
        agent_x, agent_y = int(self.agent_pos[0]), int(self.agent_pos[1])
        if self.visited_cells[agent_y, agent_x]:
            return -0.05  # Small penalty for revisiting
        return 0.0
    
    def _calculate_collision_penalty(self) -> float:
        """Calculate penalty for collision (not applicable in simple environment)."""
        # No obstacles in simple environment
        return 0.0
    
    def step(self, action):
        self.step_count += 1
        
        # Store action history
        self.action_history.append(action)
        if len(self.action_history) > 10:  # Keep only recent actions
            self.action_history = self.action_history[-10:]
        
        # Store old position
        old_pos = self.agent_pos.copy()
        
        # Process the action
        if action == 0:  # Go Forward
            # Move in current direction
            dx, dy = self.directions[self.current_direction]
            new_pos = self.agent_pos + [dx, dy]
            
            # Boundary checking
            new_pos[0] = np.clip(new_pos[0], 0, self.width - 1)
            new_pos[1] = np.clip(new_pos[1], 0, self.height - 1)
            
            # If position changed, mark as visited
            if not np.array_equal(self.agent_pos, new_pos):
                self.agent_pos = new_pos
                agent_x, agent_y = int(self.agent_pos[0]), int(self.agent_pos[1])
                if 0 <= agent_x < self.width and 0 <= agent_y < self.height:
                    self.visited_cells[agent_y, agent_x] = True
        elif action == 1:  # Turn Left
            # Change direction counter-clockwise
            self.current_direction = (self.current_direction - 1) % 4
        elif action == 2:  # Turn Right
            # Change direction clockwise
            self.current_direction = (self.current_direction + 1) % 4
        elif action == 3:  # U-turn
            # Turn 180 degrees
            self.current_direction = (self.current_direction + 2) % 4
                
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
        collision_penalty = 0.0
        if np.array_equal(old_pos, self.agent_pos) and action == 0:  # Tried to move forward but couldn't
            collision_penalty = -1.0
            
        # NEW: Data scarcity reward
        data_scarcity_reward = self._calculate_data_scarcity_reward()
        
        # NEW: Coverage reward
        coverage_reward = self._calculate_coverage_reward()
        
        # NEW: Path efficiency penalty
        path_efficiency_penalty = self._calculate_path_efficiency_penalty()
        
        # NEW: Repeated path penalty
        repeated_path_penalty = self._calculate_repeated_path_penalty()
        
        # Total reward - only using specified components
        reward = (distance_reward + time_penalty + goal_reward + collision_penalty + 
                 data_scarcity_reward + coverage_reward + path_efficiency_penalty + 
                 repeated_path_penalty)
        
        # Check if max steps reached
        if self.step_count >= self.max_steps:
            done = True
            
        info = {
            'distance_to_goal': new_distance,
            'distance_reward': distance_reward,
            'goal_reward': goal_reward,
            'collision_penalty': collision_penalty,
            'data_scarcity_reward': data_scarcity_reward,
            'coverage_reward': coverage_reward,
            'path_efficiency_penalty': path_efficiency_penalty,
            'repeated_path_penalty': repeated_path_penalty,
            'current_direction': self.current_direction
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
            
        # Draw visited cells (coverage)
        for y in range(self.height):
            for x in range(self.width):
                if self.visited_cells[y, x]:
                    rect = patches.Rectangle((x-0.5, y-0.5), 1, 1, 
                                           linewidth=0, facecolor='lightblue', alpha=0.3)
                    ax.add_patch(rect)
                    
        # Draw agent with direction indicator
        agent_circle = plt.Circle(self.agent_pos, 0.3, color='blue')
        ax.add_patch(agent_circle)
        
        # Draw direction indicator
        dir_dx, dir_dy = self.directions[self.current_direction]
        plt.arrow(self.agent_pos[0], self.agent_pos[1], dir_dx*0.3, dir_dy*0.3, 
                 head_width=0.1, head_length=0.1, fc='red', ec='red')
        
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