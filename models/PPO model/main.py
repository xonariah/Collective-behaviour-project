import gymnasium as gym
from stable_baselines3 import PPO
from PredatorPreyEnv import PredatorPreyEnv
import supersuit as ss
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

def evaluateModel(name, obstacles=[], save_animation=False):
    env = PredatorPreyEnv(num_preys=20, num_predators=4, num_super_predators=2, obstacles=obstacles)
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
        for _ in range(500):
            actions = model.predict(obs, deterministic=True)
            obs, rewards, done, info = env.step(actions[0])  # Get the actions and step the environment
            observations.append(obs)
        _save_animation(np.array(observations), 20, 4, 2, obstacles)
    else:
        for _ in range(500):
            actions = model.predict(obs, deterministic=True)
            obs, rewards, done, info = env.step(actions[0])  # Get the actions and step the environment
            env.render()

def trainNewModel(name, obstacles=[]):
    env = PredatorPreyEnv(num_preys=20, num_predators=4, num_super_predators=2, obstacles=obstacles)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, base_class="stable_baselines3")

    # Create the model
    model = PPO("MlpPolicy", env, verbose=1, n_steps=2048, learning_rate=0.001)

    # Train the model
    print("Training the model...")
    model.learn(total_timesteps=1_000)
    print("Training finished.")

    # Save the trained model
    model.save(f"models/ppo_preypredator_model_{name}")

def trainOldModel(oldName, newName, obstacles=[]):
    env = PredatorPreyEnv(num_preys=20, num_predators=4, num_super_predators=2, obstacles=obstacles)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, base_class="stable_baselines3")

    # Load the model
    model = PPO.load(f"models/ppo_preypredator_model_{oldName}", learning_rate=0.001)
    model.set_env(env)

    # Train the model
    print("Training the model...")
    model.learn(total_timesteps=200_000)
    print("Training finished.")

    # Save the trained model
    model.save(f"models/ppo_preypredator_model_{newName}")

def bareEnvironment(save_animation=False, obstacles=[]):
    env = PredatorPreyEnv(num_preys=20, num_predators=4, num_super_predators=2, obstacles=obstacles)
    obs, _ = env.reset()
    if save_animation:
        observations = []
        temp = []
        for values in obs.values():
            temp.append(list(values))
        observations.append(temp)
        for _ in range(500):
            actions = {}
            # Sample actions for all agents
            for agent in env.agents:
                actions[agent] = env.action_spaces[agent].sample()  # Sample action for each agent

            # Step the environment with the sampled actions
            obs, rewards, done, terminated, info = env.step(actions)
            temp = []
            for values in obs.values():
                temp.append(list(values))
            observations.append(temp)
        _save_animation(np.array(observations), 20, 4, 2, env.obstacles)
    else:
        for _ in range(500):
            actions = {}

            # Sample actions for all agents
            for agent in env.agents:
                actions[agent] = env.action_spaces[agent].sample()  # Sample action for each agent

            # Step the environment with the sampled actions
            obs, rewards, done, terminated, info = env.step(actions)

            env.render()


def _save_animation(observation, num_prey, num_predator, num_super_predator, obstacles, filename="animation.gif"):
    """
    Saves an animation of the agents' positions based on the observation data.
    """
    
    timesteps = observation.shape[0]
    
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-10, 10)  # Adjust limits based on the expected position range
    ax.set_ylim(-10, 10)
    
    scat = ax.scatter([], [], s=50)
    lines = []  # List to store line objects

    for (x, y, radius) in obstacles:
        circle = patches.Circle((x, y), radius=radius, edgecolor='grey', facecolor='grey')
        ax.add_patch(circle)
    
    def update(frame):
        """
        Update function for each frame in the animation.
        """
        # Extract positions and orientations for the current frame
        positions = observation[frame, :, :2]  # Get x, y positions
        angles = observation[frame, :, 4]  # Get orientation angles (if needed)

        colors = ['blue'] * num_prey + ['orange'] * num_predator + ['red'] * num_super_predator
        
        scat.set_offsets(positions)
        scat.set_color(colors)
        
        # Draw or update lines for orientations
        for i, (pos, orient) in enumerate(zip(positions, angles)):
            # Calculate the end position of the line based on orientation
            end_x = pos[0] + np.cos(orient) * 0.5  # Adjust length as needed
            end_y = pos[1] + np.sin(orient) * 0.5  # Adjust length as needed

            # Choose color based on the agent index
            line_color = colors[i]

            # Create new line or update existing ones
            if len(lines) <= i:
                line, = ax.plot([pos[0], end_x], [pos[1], end_y], c=line_color)
                lines.append(line)
            else:
                lines[i].set_data([pos[0], end_x], [pos[1], end_y])
                lines[i].set_color(line_color)  # Update color if needed
        
        return scat, *lines

    # Create the animation
    ani = FuncAnimation(fig, update, frames=timesteps, blit=True)
    
    ani.save(filename, fps=30)

    print(f"Animation saved as {filename}")

if __name__ == "__main__":
    obstacles = [(0,0,2), (3,2,3), (-4,-8,1), (-6,5,2)]
    bareEnvironment(obstacles=obstacles)
    #trainNewModel(0, obstacles=obstacles)
    """for i in range(10):
        print(f"Episode {i}")
        trainOldModel(i, i+1, obstacles=obstacles)"""