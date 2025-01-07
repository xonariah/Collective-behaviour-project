import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class PreyPredatorEnv(gym.Env):
    """
    Custom Environment with Prey and Predator agents, compatible with Gymnasium.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, num_preys, num_predators, delta_t=0.1, C=0.3):
        super(PreyPredatorEnv, self).__init__()

        # Define the number of agents
        self.num_preys = num_preys
        self.num_predators = num_predators
        self.num_agents = self.num_preys + self.num_predators

        self.delta_t = delta_t # Time step
        self.C = C # Drag coefficient

        # Define the environment boundaries
        self.x_limit = 10.0
        self.y_limit = 10.0

        # Observation space: periodic positions (x, y), velocity (vx, vy), and orientation (theta) for each agent
        self.observation_space = spaces.Box(
            low=np.tile([-self.x_limit, -self.y_limit, -np.inf, -np.inf, -np.pi], (self.num_agents, 1)),
            high=np.tile([self.x_limit, self.y_limit, np.inf, np.inf, np.pi], (self.num_agents, 1)),
            dtype=np.float64
        )

        # Action space: rotational force and propulsion force for each agent
        self.action_space = spaces.Box(
            low=np.tile([-1.0, 0.0], (self.num_agents, 1)), 
            high=np.tile([1.0, 1.0], (self.num_agents, 1)), 
            dtype=np.float64
        )

        # Initialize the state
        self.state = None

    def reset(self, seed=None):
        """
        Reset the environment to an initial state.
        """
        if seed is not None:
            np.random.seed(seed)

        self.state = np.zeros((self.num_agents, 5), dtype=np.float32)

        # Initialize positions randomly within the environment bounds
        self.state[:, 0] = np.random.uniform(-self.x_limit, self.x_limit, size=self.num_agents)  # x positions
        self.state[:, 1] = np.random.uniform(-self.y_limit, self.y_limit, size=self.num_agents)  # y positions

        # Initialize velocities to zero
        self.state[:, 2:4] = 0.0

        # Initialize orientations randomly
        self.state[:, 4] = np.random.uniform(-np.pi, np.pi, size=self.num_agents)  # theta

        return self.state, {} # Only return the state and info (empty dict)

    def step(self, actions):
        """
        Execute one step in the environment.

        :param actions: Array of actions for all agents. Each action has two components: rotational and propulsion forces.
        :return: (state, reward, done, info)
        """
        assert actions.shape == (self.num_agents, 2), "Invalid action shape!"

        # Update the state for each agent
        for i in range(self.num_agents):
            rotation_force, propulsion_force = actions[i]

            # Update orientation
            self.state[i, 4] += rotation_force * self.delta_t

            v = self.state[i, 2]**2 + self.state[i, 3]**2
            if v > 0:   # Using if to avoid division by zero
                # Calculate drag force based on velocity
                drag_force = -self.C * np.sqrt(v)
                # Update velocity based on propulsion force, orientation, and drag force
                self.state[i, 2] += (propulsion_force * np.cos(self.state[i, 4]) + drag_force * self.state[i, 2] / v) * self.delta_t # vx
                self.state[i, 3] += (propulsion_force * np.sin(self.state[i, 4]) + drag_force * self.state[i, 3] / v) * self.delta_t # vy
            else:
                # Update velocity based on propulsion force and orientation
                self.state[i, 2] += propulsion_force * np.cos(self.state[i, 4]) * self.delta_t # vx
                self.state[i, 3] += propulsion_force * np.sin(self.state[i, 4]) * self.delta_t # vy


        # Update positions based on velocity
        self.state[:, 0] += self.state[:, 2] * self.delta_t # x positions
        self.state[:, 1] += self.state[:, 3] * self.delta_t # y positions

        # Apply boundary conditions (agents wrap around the edges)
        self.state[:, 0] = np.mod(self.state[:, 0] + self.x_limit, 2 * self.x_limit) - self.x_limit
        self.state[:, 1] = np.mod(self.state[:, 1] + self.y_limit, 2 * self.y_limit) - self.y_limit

        # Calculate rewards (example logic: predators aim to minimize distance to preys, preys aim to maximize it)
        rewards = np.zeros(self.num_agents)
        prey_positions = self.state[:self.num_preys, :2]
        predator_positions = self.state[self.num_preys:, :2]

        for i in range(self.num_preys):
            distances = np.linalg.norm(predator_positions - prey_positions[i], axis=1)
            rewards[i] = np.min(distances)  # Preys maximize distance from predators

        for i in range(self.num_predators):
            distances = np.linalg.norm(prey_positions - predator_positions[i], axis=1)
            rewards[self.num_preys + i] = -np.min(distances)  # Predators minimize distance to preys
        
        
        # Check for termination condition (example: fixed number of steps or collision)
        done = False
        terminated = False # (For now just False) Example termination condition: all preys are caught

        # Additional information
        info = {}

        return self.state, rewards, done, terminated, info

    def render(self, mode="human"):
        """
        Render the environment (animation).
        """
        if not hasattr(self, 'fig'):
            self.fig, self.ax = plt.subplots(figsize=(6, 6))
            self.ax.set_xlim(-self.x_limit, self.x_limit)
            self.ax.set_ylim(-self.y_limit, self.y_limit)

            # Scatter plots for preys and predators
            self.prey_scatter = self.ax.scatter([], [], c='blue', label='Preys')
            self.predator_scatter = self.ax.scatter([], [], c='orange', label='Predators')

        prey_positions = self.state[:self.num_preys, :2]
        predator_positions = self.state[self.num_preys:, :2]

        self.prey_scatter.set_offsets(prey_positions)
        self.predator_scatter.set_offsets(predator_positions)

        plt.pause(0.01)

    def close(self):
        """
        Clean up resources (if any).
        """
        if hasattr(self, 'fig'):
            plt.close(self.fig)


