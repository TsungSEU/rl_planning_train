#!/usr/bin/env python3
"""
Demo script for cloud training with both logits and softmax outputs
This demonstrates how to use the model with both softmax for metrics calculation 
and logits for training during cloud training
"""

import torch
import numpy as np
from torch.distributions import Categorical
import sys
import os

# Add parent directory to path to import environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from planner_train import ActorCritic


def demo_cloud_training_usage():
    """
    Demonstrate how to use the ActorCritic model during cloud training
    with both logits for training and softmax for metrics calculation
    """
    print("Cloud Training Demo - Using both logits and softmax")
    print("=" * 50)
    
    # Initialize model
    state_dim = 24  # As per our environment
    action_dim = 4  # Forward, Turn Left, Turn Right, U-turn
    model = ActorCritic(state_dim=state_dim, action_dim=action_dim)
    
    # Simulate a state tensor
    state = torch.randn(1, state_dim)  # Batch size of 1
    
    print("1. Using logits for training (default behavior):")
    logits, value = model(state, use_softmax=False)  # Default behavior
    print(f"   Logits shape: {logits.shape}")
    print(f"   Logits values: {logits[0].detach().numpy()}")
    print(f"   Value: {value.item():.4f}")
    
    # Use logits for training (with Categorical distribution)
    dist = Categorical(logits=logits)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    print(f"   Sampled action: {action.item()}")
    print(f"   Log probability: {log_prob.item():.4f}")
    
    print("\n2. Using softmax for metrics calculation during cloud training:")
    softmax_probs, value = model(state, use_softmax=True)  # For metrics calculation
    print(f"   Softmax probabilities shape: {softmax_probs.shape}")
    print(f"   Softmax probabilities: {softmax_probs[0].detach().numpy()}")
    print(f"   Value: {value.item():.4f}")
    
    # Verify that probabilities sum to 1 (approximately)
    prob_sum = softmax_probs.sum().item()
    print(f"   Sum of probabilities: {prob_sum:.6f}")
    
    print("\n3. Action selection using softmax (deterministic or sampling):")
    # Deterministic: take action with highest probability
    deterministic_action = torch.argmax(softmax_probs, dim=-1).item()
    print(f"   Deterministic action (argmax): {deterministic_action}")
    
    # Stochastic: sample from the probability distribution
    sampled_action = torch.multinomial(softmax_probs.squeeze(), 1).item()
    print(f"   Stochastic action (sampled): {sampled_action}")
    
    print("\n4. Metrics that can be calculated with softmax probabilities:")
    # Entropy of the policy (measure of randomness)
    entropy = -(softmax_probs * torch.log(softmax_probs + 1e-8)).sum(dim=-1).item()
    print(f"   Policy entropy: {entropy:.4f}")
    
    # Maximum probability (measure of confidence)
    max_prob = torch.max(softmax_probs, dim=-1)[0].item()
    print(f"   Max probability (confidence): {max_prob:.4f}")
    
    # Number of effective actions (inverse of squared probabilities)
    effective_actions = 1.0 / (softmax_probs ** 2).sum().item()
    print(f"   Effective number of actions: {effective_actions:.4f}")
    
    print("\nThis demonstrates how during cloud training:")
    print("- Use logits (use_softmax=False) for training with Categorical distribution")
    print("- Use softmax (use_softmax=True) for calculating metrics and monitoring")
    print("- .pt model files save only network parameters, not specific outputs")


def demo_training_vs_metrics():
    """
    Show the difference between using logits for training vs softmax for metrics
    """
    print("\n" + "=" * 50)
    print("Training vs Metrics Usage Comparison")
    print("=" * 50)
    
    # Initialize model
    state_dim = 24
    action_dim = 4
    model = ActorCritic(state_dim=state_dim, action_dim=action_dim)
    
    # Simulate a state
    state = torch.randn(1, state_dim)
    
    # Training usage (logits)
    print("Training Usage (logits):")
    logits, value = model(state, use_softmax=False)
    dist = Categorical(logits=logits)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    loss = -log_prob * 0.5  # Example loss calculation
    print(f"  Action: {action.item()}, LogProb: {log_prob.item():.4f}, Loss: {loss.item():.4f}")
    
    # Metrics usage (softmax)
    print("\nMetrics Usage (softmax):")
    softmax_probs, value = model(state, use_softmax=True)
    entropy = -(softmax_probs * torch.log(softmax_probs + 1e-8)).sum(dim=-1).item()
    confidence = torch.max(softmax_probs, dim=-1)[0].item()
    print(f"  Entropy: {entropy:.4f}, Confidence: {confidence:.4f}")
    

if __name__ == "__main__":
    demo_cloud_training_usage()
    demo_training_vs_metrics()