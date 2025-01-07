import gymnasium as gym
from stable_baselines3 import PPO
from PredatorPreyEnv import PredatorPreyEnv
import supersuit as ss

def evaluateModel(name):
    env = PredatorPreyEnv(num_preys=20, num_predators=3)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, base_class="stable_baselines3")

    # Load the model
    model = PPO.load(f"models/ppo_preypredator_model_{name}")

    # Evaluate the model in the environment
    obs = env.reset()
    done = False
    for _ in range(500):
        actions = model.predict(obs, deterministic=True)
        state, rewards, done, info = env.step(actions[0])  # Get the actions and step the environment
        env.render()

def trainNewModel(name):
    env = PredatorPreyEnv(num_preys=20, num_predators=3)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, base_class="stable_baselines3")

    # Create the model
    model = PPO("MlpPolicy", env, verbose=1, n_steps=2048)

    # Train the model
    print("Training the model...")
    model.learn(total_timesteps=1_000_000)
    print("Training finished.")

    # Save the trained model
    model.save(f"models/ppo_preypredator_model_{name}")

    # Evaluate the model in the environment
    obs = env.reset()
    done = False
    for _ in range(500):
        actions = model.predict(obs, deterministic=True)
        state, rewards, done, info = env.step(actions[0])  # Get the actions and step the environment
        env.render()

def trainOldModel(oldName, newName):
    env = PredatorPreyEnv(num_preys=20, num_predators=3)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, base_class="stable_baselines3")

    # Load the model
    model = PPO.load(f"models/ppo_preypredator_model_{oldName}")
    model.set_env(env)

    # Train the model
    print("Training the model...")
    model.learn(total_timesteps=1_000_000)
    print("Training finished.")

    # Save the trained model
    model.save(f"models/ppo_preypredator_model_{newName}")

def bareEnvironment():
    env = PredatorPreyEnv(num_preys=20, num_predators=3)
    for _ in range(200):
        actions = {}

        # Sample actions for all agents
        for agent in env.agents:
            actions[agent] = env.action_spaces[agent].sample()  # Sample action for each agent

        # Step the environment with the sampled actions
        state, rewards, done, terminated, info = env.step(actions)

        env.render()


if __name__ == "__main__":
    evaluateModel(5)
    """for i in range(5, 100):
        print(f"Episode {i}")
        trainOldModel(i, i+1)"""