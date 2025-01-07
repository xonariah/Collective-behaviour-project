import gymnasium as gym
from stable_baselines3 import PPO
from PredatorPreyEnv import PredatorPreyEnv
import supersuit as ss

train = True
use_model = True

if use_model:
    # Create environment instance
    env = PredatorPreyEnv(num_preys=10, num_predators=2)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, base_class="stable_baselines3")

    if train:
        # Create the model
        model = PPO("MlpPolicy", env, verbose=1, n_steps=512, batch_size=64)

        # Train the model
        print("Training the model...")
        model.learn(total_timesteps=40000)
        print("Training finished.")

        # Save the trained model
        model.save("models/ppo_preypredator_model")
    else:
        model = PPO.load("models/ppo_preypredator_model")

    # Evaluate the model in the environment
    obs = env.reset()
    done = False
    for _ in range(500):
        actions = model.predict(obs, deterministic=True)
        state, rewards, done, info = env.step(actions[0])  # Get the actions and step the environment
        env.render()
else:
    env = PredatorPreyEnv(num_preys=10, num_predators=2)
    for _ in range(200):
        actions = {}

        # Sample actions for all agents
        for agent in env.agents:
            actions[agent] = env.action_spaces[agent].sample()  # Sample action for each agent

        # Step the environment with the sampled actions
        state, rewards, done, terminated, info = env.step(actions)

        env.render()