#!/usr/bin/env python3
"""
Visualize the learning results of the trained model
Shows successful path planning examples
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import sys
import os

# Add parent directory to path to import environment
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment import SimplePathPlanningEnv


def visualize_path(env, path, title="Path Visualization"):
    """
    Visualize a path in the environment.
    """
    # Dynamically adjust figure size based on map size
    fig_size = max(10, min(20, env.width // 3, env.height // 3))
    fig, ax = plt.subplots(1, 1, figsize=(fig_size, fig_size))
    
    # Draw agent path with gradient colors to show progression
    path_array = np.array(path)
    
    # Create a color map for the path to show progression
    colors = plt.cm.viridis(np.linspace(0, 1, len(path_array)))
    
    # Plot each segment with a different color
    for i in range(len(path_array)-1):
        ax.plot(path_array[i:i+2, 0], path_array[i:i+2, 1], 
                color=colors[i], marker='o', markersize=4, linewidth=2)
    
    # Mark start and goal with distinct markers
    start = path[0]
    goal = path[-1]
    
    # Draw start point (green triangle)
    ax.plot(start[0], start[1], 'g^', markersize=15, markeredgecolor='black', markeredgewidth=1, label='Start')
    
    # Draw goal point (red star)
    ax.plot(goal[0], goal[1], 'r*', markersize=20, markeredgecolor='black', markeredgewidth=1, label='Goal')
    
    # Formatting
    ax.set_xlim(-1, env.width)
    ax.set_ylim(-1, env.height)
    ax.set_aspect('equal')
    
    # For large maps, reduce grid density for better visibility
    if env.width > 50:
        # For very large maps, use sparse grid
        major_ticks = np.arange(0, env.width, 10)
        minor_ticks = np.arange(0, env.width, 2)
        ax.set_xticks(major_ticks)
        ax.set_yticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)
        ax.set_yticks(minor_ticks, minor=True)
        ax.grid(which='major', color='gray', linestyle='-', linewidth=1.0)
        ax.grid(which='minor', color='lightgray', linestyle=':', linewidth=0.5)
    elif env.width > 30:
        # For medium-large maps
        major_ticks = np.arange(0, env.width, 5)
        minor_ticks = np.arange(0, env.width, 1)
        ax.set_xticks(major_ticks)
        ax.set_yticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)
        ax.set_yticks(minor_ticks, minor=True)
        ax.grid(which='major', color='gray', linestyle='-', linewidth=0.8)
        ax.grid(which='minor', color='lightgray', linestyle=':', linewidth=0.5)
    else:
        # For small maps
        ax.grid(True, color='gray', linestyle='-', linewidth=0.5)
        ax.set_xticks(range(env.width))
        ax.set_yticks(range(env.height))
    
    ax.set_title(title, fontsize=14, pad=20)
    ax.legend(loc='upper right', fontsize=12)
    
    return fig


def main():
    # Load evaluation results
    with open('final_evaluation.json', 'r') as f:
        results = json.load(f)
    
    # Show successful cases
    successful_cases = [k for k, v in results.items() 
                       if k != 'summary' and v['success']]
    
    print(f"Found {len(successful_cases)} successful cases")
    
    # Counter for saved visualizations
    viz_counter = 0
    
    # Group cases by distance for better visualization selection
    short_cases = []
    medium_cases = []
    long_cases = []
    very_long_cases = []
    
    for case_key in successful_cases:
        case = results[case_key]
        start, goal = case['start'], case['goal']
        distance = ((goal[0] - start[0])**2 + (goal[1] - start[1])**2)**0.5
        
        if distance <= 5:
            short_cases.append(case_key)
        elif distance <= 15:
            medium_cases.append(case_key)
        elif distance <= 30:
            long_cases.append(case_key)
        else:
            very_long_cases.append(case_key)
    
    # Select representative cases from each group
    cases_to_show = []
    cases_to_show.extend(short_cases[:2])      # Show 2 short cases
    cases_to_show.extend(medium_cases[:2])     # Show 2 medium cases
    cases_to_show.extend(long_cases[:2])       # Show 2 long cases
    cases_to_show.extend(very_long_cases[:2])  # Show 2 very long cases
    
    print(f"Showing {len(cases_to_show)} representative cases")
    
    for case_key in cases_to_show:
        case = results[case_key]
        print(f"\nCase: {case['start']} -> {case['goal']}")
        print(f"  Steps: {case['steps']}")
        print(f"  Reward: {case['reward']:.2f}")
        
        # Determine map size based on start/goal positions
        max_coord = max(case['start'][0], case['start'][1], case['goal'][0], case['goal'][1])
        if max_coord >= 100:
            map_width = 101
            map_height = 101
        elif max_coord >= 50:
            map_width = 51
            map_height = 51
        else:
            map_width = 20
            map_height = 20
            
        print(f"  Map size: {map_width}x{map_height}")
        
        # Create environment for visualization
        env = SimplePathPlanningEnv(width=map_width, height=map_height)
        env.reset(start_pos=case['start'], goal_pos=case['goal'])
        
        # Visualize path
        path = case['path']
        fig = visualize_path(env, path, 
                           f"Path: {case['start']} → {case['goal']} | "
                           f"Steps: {case['steps']} | Reward: {case['reward']:.2f}")
        
        # Save visualization
        viz_counter += 1
        filename = f'learned_path_{viz_counter}.png'
        plt.savefig(filename, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved visualization to {filename}")
    
    # Also visualize some failed cases to understand failure modes
    failed_cases = [k for k, v in results.items() 
                   if k != 'summary' and not v['success']]
    
    print(f"\nFound {len(failed_cases)} failed cases")
    
    # Show up to 3 failed cases
    for case_key in failed_cases[:3]:
        case = results[case_key]
        print(f"\nFailed Case: {case['start']} -> {case['goal']}")
        print(f"  Steps: {case['steps']}")
        print(f"  Final distance to goal: {case['final_distance']:.2f}")
        print(f"  Reward: {case['reward']:.2f}")
        
        # Determine map size
        max_coord = max(case['start'][0], case['start'][1], case['goal'][0], case['goal'][1])
        if max_coord >= 100:
            map_width = 101
            map_height = 101
        elif max_coord >= 50:
            map_width = 51
            map_height = 51
        else:
            map_width = 20
            map_height = 20
            
        print(f"  Map size: {map_width}x{map_height}")
        
        # Create environment for visualization
        env = SimplePathPlanningEnv(width=map_width, height=map_height)
        env.reset(start_pos=case['start'], goal_pos=case['goal'])
        
        # Visualize path
        path = case['path']
        fig = visualize_path(env, path, 
                           f"Failed Path: {case['start']} → {case['goal']} | "
                           f"Steps: {case['steps']} | Final Dist: {case['final_distance']:.2f}")
        
        # Save visualization
        viz_counter += 1
        filename = f'learned_path_{viz_counter}.png'
        plt.savefig(filename, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved visualization to {filename}")
    
    # Print summary
    summary = results['summary']
    print(f"\nSummary:")
    print(f"  Successful Cases: {summary['successful_cases']}")
    print(f"  Total Cases: {summary['total_cases']}")
    print(f"  Success Rate: {summary['success_rate']:.2%}")
    
    print("\nModel successfully learned to navigate in simpler environments!")
    print("For more complex tasks (longer distances), the model may need more training or architectural improvements.")


if __name__ == "__main__":
    main()