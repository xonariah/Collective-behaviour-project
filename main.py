import gymnasium as gym
from stable_baselines3 import PPO
from PredatorPreyEnv import PredatorPreyEnv
import supersuit as ss
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def evaluateModel(name, save_animation=False):
    env = PredatorPreyEnv(num_preys=20, num_predators=3)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, base_class="stable_baselines3")

    # Load the model
    model = PPO.load(f"models/ppo_preypredator_model_{name}")

    # Evaluate the model in the environment
    obs = env.reset()
    done = False
    if save_animation:
        observations = []
        observations.append(obs)
        for _ in range(200):
            actions = model.predict(obs, deterministic=True)
            obs, rewards, done, info = env.step(actions[0])  # Get the actions and step the environment
            observations.append(obs)
        _save_animation(np.array(observations))
    else:
        for _ in range(500):
            actions = model.predict(obs, deterministic=True)
            obs, rewards, done, info = env.step(actions[0])  # Get the actions and step the environment
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
    for _ in range(500):
        actions = {}

        # Sample actions for all agents
        for agent in env.agents:
            actions[agent] = env.action_spaces[agent].sample()  # Sample action for each agent

        # Step the environment with the sampled actions
        obs, rewards, done, terminated, info = env.step(actions)

        env.render()


def _save_animation(observation, filename="animation.gif"):
    """
    Saves an animation of the agents' positions based on the observation data.
    """
    
    timesteps = observation.shape[0]
    
    # Set up the figure and axis
    fig, ax = plt.subplots()
    ax.set_xlim(-10, 10)  # Adjust limits based on the expected position range
    ax.set_ylim(-10, 10)
    
    scat = ax.scatter([], [], s=50)
    
    def update(frame):
        """
        Update function for each frame in the animation.
        """
        # Extract positions and orientations for the current frame
        positions = observation[frame, :, :2]  # Get x, y positions
        angles = observation[frame, :, 4]  # Get orientation angles (if needed)

        colors = ['blue'] * 20 + ['orange'] * 3
        
        scat.set_offsets(positions)
        scat.set_color(colors)
        
        
        return scat,

    # Create the animation
    ani = FuncAnimation(fig, update, frames=timesteps, blit=True)
    
    ani.save(filename, fps=30)

    print(f"Animation saved as {filename}")

if __name__ == "__main__":
    evaluateModel(47, save_animation=True)
    """for i in range(11, 200):
        print(f"Episode {i}")
        trainOldModel(i, i+1)"""