
import gymnasium as gym
import rware
import numpy as np

env = gym.make("rware-tiny-2ag-v2")

obs, info = env.reset(seed=42)

print("Observation space:", env.observation_space)
print("Observation shape:", np.shape(obs))

print("Action space:", env.action_space)
print("Action space shape:", env.action_space.shape)
print("Number of robots:", env.unwrapped.n_agents)

print("Sample observation:", obs)
print("Sample action:", env.action_space.sample())

env.close()
