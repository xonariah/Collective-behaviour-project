import gymnasium as gym
from stable_baselines3 import PPO
from PreyPredatorEnv import PreyPredatorEnv

# Initialize the environment
env = PreyPredatorEnv(num_preys=8, num_predators=2)

# Test the trained model
state = env.reset()
for _ in range(500):
    actions = env.action_space.sample()
    state, rewards, done, terminated, info = env.step(actions)
    env.render()

env.close()