import numpy as np
from pettingzoo.utils.env import ParallelEnv
from gymnasium import spaces
import matplotlib.pyplot as plt

class PredatorPreyEnv(ParallelEnv):
    metadata = {
        "render.modes": ["human"],
        "name": "predator_prey_v0",
    }

    def __init__(self, num_preys=4, num_predators=2, render_mode=None, delta_t=0.1, C=0.2):
        super().__init__()
        
        self.num_preys = num_preys
        self.num_predators = num_predators
        self.num_a = self.num_preys + self.num_predators
        
        self.possible_agents = [f"prey_{i}" for i in range(self.num_preys)] + [f"predator_{i}" for i in range(self.num_predators)]
        self.agents = self.possible_agents[:]
        self.agent_types = {agent: "prey" if "prey" in agent else "predator" for agent in self.agents}
        
        # Define the environment boundaries
        self.x_limit = 10.0
        self.y_limit = 10.0

        self.delta_t = delta_t # Time step
        self.C = C # Drag coefficient
        
        self.observation_spaces = {agent: self._get_observation_space() for agent in self.possible_agents}
        self.action_spaces = {agent: self._get_action_space() for agent in self.possible_agents}

        self.render_mode = render_mode

        self.reset()
    
    def observation_space(self, agent):
        return self.observation_spaces[agent]
    
    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, return_info=False, options=None):
        # Initialize positions and velocities of agents
        self.positions = {
            agent: np.array([np.random.uniform(-self.x_limit, self.x_limit), np.random.uniform(-self.y_limit, self.y_limit)])
            for agent in self.agents
        }

        self.velocities = {agent: np.zeros(2) for agent in self.agents}
        self.orientations = {agent: np.random.uniform(-np.pi, np.pi) for agent in self.agents}
        
        # Initialize done and rewards
        self._dones = {agent: False for agent in self.agents}
        self._rewards = {agent: 0.0 for agent in self.agents}

        observations = {agent: self._get_obs(agent) for agent in self.agents}

        # Create the 'infos' dictionary, which can be empty or contain additional info
        infos = {agent: {} for agent in self.agents}  # You can add more info here if needed

        return observations, infos

    def step(self, actions):
        # Apply actions
        for agent, action in actions.items():
            rotation_force, propulsion_force = action
            
            # Update orientation
            self.orientations[agent] += rotation_force * self.delta_t
            
            v = np.linalg.norm(self.velocities[agent])
            if v > 0:
                drag_force = -self.C * self.velocities[agent]

                forward_vector = np.array([np.cos(self.orientations[agent]), np.sin(self.orientations[agent])])
                self.velocities[agent] += (forward_vector * propulsion_force + drag_force) * self.delta_t
            else:
                forward_vector = np.array([np.cos(self.orientations[agent]), np.sin(self.orientations[agent])])
                self.velocities[agent] += forward_vector * propulsion_force * self.delta_t

        # Update positions
        for agent in self.agents:
            self.positions[agent] += self.velocities[agent] * self.delta_t
            
            # Apply periodic boundary conditions (wrap around the arena)
            self.positions[agent][0] = np.mod(self.positions[agent][0] + self.x_limit, 2 * self.x_limit) - self.x_limit
            self.positions[agent][1] = np.mod(self.positions[agent][1] + self.y_limit, 2 * self.y_limit) - self.y_limit

        # Check interactions and update rewards
        self._rewards = self._get_rewards(actions)

        # Initialize done flags for each agent
        self._dones = {agent: False for agent in self.agents}
        
        # Check if all agents are done
        dones = {agent: self._dones[agent] for agent in self.agents}
        dones["__all__"] = all(self._dones.values())

        # Termination and truncation: no specific conditions in your environment
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}

        # Get observations and infos
        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        return observations, self._rewards, terminations, truncations, infos

    def _periodic_distance(self, agent1, agent2):
            diff = self.positions[agent1] - self.positions[agent2]
            # Apply periodic boundary by adjusting the difference to wrap around
            diff[0] -= np.round(diff[0] / (2 * self.x_limit)) * 2 * self.x_limit
            diff[1] -= np.round(diff[1] / (2 * self.y_limit)) * 2 * self.y_limit
            return np.linalg.norm(diff)

    def _get_obs(self, agent):
        # Get relative positions and velocities of all agents
        agent_pos = self.positions[agent]
        agent_vel = self.velocities[agent]
        agent_angle = self.orientations[agent]
        
        return np.array([*agent_pos, *agent_vel, agent_angle], dtype=np.float32)
    
    def _get_rewards(self, actions):
        rewards = {agent: 0.0 for agent in self.agents}

        kill_distance = 0.2  # predator-prey kill distance threshold

        for predator in [agent for agent in self.agents if self.agent_types[agent] == "predator"]:
            for prey in [agent for agent in self.agents if self.agent_types[agent] == "prey"]:
                distance = self._periodic_distance(predator, prey)
                
                # Reward predator for kill and penalize prey
                if distance < kill_distance:
                    rewards[predator] += 1.0
                    rewards[prey] -= 1.0

                # Reward prey for being far from predators
                #rewards[prey] += (1 - np.exp(-distance)) * 0.1 / self.num_predators

                # Penalize predator for being far from preys
                #rewards[predator] -= (1 - np.exp(-distance)) * 0.1 / self.num_preys

        for agent, action in actions.items():
            rotation_force, propulsion_force = action
            rewards[agent] -= 0.1 * np.abs(rotation_force)
            rewards[agent] -= 0.01 * np.abs(propulsion_force)
        
        return rewards

    def render(self):
        if not hasattr(self, 'fig'):
            self.fig, self.ax = plt.subplots(figsize=(6, 6))
            self.ax.set_xlim(-self.x_limit, self.x_limit)
            self.ax.set_ylim(-self.y_limit, self.y_limit)

            # Scatter plots for preys and predators
            self.prey_scatter = self.ax.scatter([], [], c='blue', label='Preys')
            self.predator_scatter = self.ax.scatter([], [], c='orange', label='Predators')

            # Create a list to store orientation lines
            self.prey_lines = []
            self.predator_lines = []

        # Clear orientation lines
        for line in self.prey_lines:
            line.remove()
        for line in self.predator_lines:
            line.remove()

        self.prey_lines = []
        self.predator_lines = []

        # Extract positions for all agents
        prey_positions = np.array([self.positions[agent] for agent in self.agents if "prey" in agent])
        predator_positions = np.array([self.positions[agent] for agent in self.agents if "predator" in agent])

        prey_orientations = np.array([self.orientations[agent] for agent in self.agents if "prey" in agent])
        predator_orientations = np.array([self.orientations[agent] for agent in self.agents if "predator" in agent])

        self.prey_scatter.set_offsets(prey_positions)
        self.predator_scatter.set_offsets(predator_positions)

        # Draw orientation lines for preys
        if len(prey_positions) > 0:
            for pos, orient in zip(prey_positions, prey_orientations):
                end_x = pos[0] + np.cos(orient) * 0.5  # Adjust the length as needed
                end_y = pos[1] + np.sin(orient) * 0.5
                line, = self.ax.plot([pos[0], end_x], [pos[1], end_y], c='blue')
                self.prey_lines.append(line)

        # Draw orientation lines for predators
        if len(predator_positions) > 0:
            for pos, orient in zip(predator_positions, predator_orientations):
                end_x = pos[0] + np.cos(orient) * 0.5  # Adjust the length as needed
                end_y = pos[1] + np.sin(orient) * 0.5
                line, = self.ax.plot([pos[0], end_x], [pos[1], end_y], c='orange')
                self.predator_lines.append(line)

        plt.pause(0.01)

    def _get_observation_space(self):
        observation_space = spaces.Box(
            low=np.array([-self.x_limit, -self.y_limit, -np.inf, -np.inf, -np.pi]), 
            high=np.array([self.x_limit, self.y_limit, np.inf, np.inf, np.pi]), 
            dtype=np.float64)
        return observation_space
    
    def _get_action_space(self):
        action_space = spaces.Box(low=np.array([-1.0, 0.0]), high=np.array([1.0, 1.0]), dtype=np.float64)
        return action_space
    