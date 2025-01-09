import numpy as np
from pettingzoo.utils.env import ParallelEnv
from gymnasium import spaces
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class PredatorPreyEnv(ParallelEnv):
    metadata = {
        "render.modes": ["human"],
        "name": "predator_prey_v0",
    }

    def __init__(self, num_preys=4, num_predators=2, num_super_predators=0, render_mode=None, delta_t=0.1, C=0.3, obstacles=[]):
        super().__init__()
        
        self.num_preys = num_preys
        self.num_predators = num_predators
        self.num_super_predators = num_super_predators
        self.num_a = self.num_preys + self.num_predators + self.num_super_predators
        
        self.possible_agents = [f"prey_{i}" for i in range(self.num_preys)] + [f"predator_{i}" for i in range(self.num_predators)] + [f"super_predator_{i}" for i in range(self.num_super_predators)]
        self.agents = self.possible_agents[:]
        # Define the type of each agent: 0 for prey, 1 for predator, 2 for super predator
        self.agent_types = {}
        for agent in self.agents:
            if "super_predator" in agent:
                self.agent_types[agent] = 2
            elif "predator" in agent:
                self.agent_types[agent] = 1
            else:
                self.agent_types[agent] = 0
        
        # Define the environment boundaries
        self.x_limit = 10.0
        self.y_limit = 10.0
        self.collision_distance = 0.5
        self.kill_distance = 0.8

        self.delta_t = delta_t # Time step
        self.C = C # Drag coefficient

        self.obstacles = obstacles # List of obstacles presented as list of (x, y, radius) tuples
        
        self.observation_spaces = {agent: self._get_observation_space() for agent in self.possible_agents}
        self.action_spaces = {agent: self._get_action_space() for agent in self.possible_agents}

        self.render_mode = render_mode

        self.reset()
    
    def observation_space(self, agent):
        return self.observation_spaces[agent]
    
    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, return_info=False, options=None):
        self.timesteps_left = 1000 # Number of timesteps before the environment terminates

        # Initialize positions and velocities of agents
        self.positions = {
            agent: np.array([np.random.uniform(-self.x_limit, self.x_limit), np.random.uniform(-self.y_limit, self.y_limit)])
            for agent in self.agents
        }

        self.velocities = {agent: np.zeros(2) for agent in self.agents}
        self.orientations = {agent: np.random.uniform(-np.pi, np.pi) for agent in self.agents}

        self.distances = np.array([[self._periodic_distance(self.positions[agent1], self.positions[agent2]) for agent2 in self.agents] for agent1 in self.agents])
        
        # Initialize rewards
        self._rewards = {agent: 0.0 for agent in self.agents}

        observations = {agent: self._get_obs(agent) for agent in self.agents}

        # Create the 'infos' dictionary, which can be empty or contain additional info
        infos = {agent: {} for agent in self.agents}  # You can add more info here if needed

        return observations, infos

    def step(self, actions):
        # Apply actions
        for agent_index, agent in enumerate(self.agents):
            action = actions[agent]
            rotation_force, propulsion_force = action

            # Collision force between agents
            for other_agent_index, other_agent in enumerate(self.agents):
                if agent == other_agent:
                    continue
                distance = self.distances[agent_index, other_agent_index]
                if distance < self.collision_distance:
                    if distance < 1e-6:
                        distance = 1e-6 # Avoid division by zero
                    collision_force = (self.positions[agent] - self.positions[other_agent]) / (distance**2) * 1.5 # Collision factor
                    magnitude = np.linalg.norm(collision_force)
                    if magnitude > 10:
                        collision_force *= 10 / magnitude
                    self.velocities[agent] += collision_force * self.delta_t

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

            # Check for collision with obstacles
            for obstacle in self.obstacles:
                distance = self._periodic_distance(self.positions[agent], obstacle[:2])
                if distance < obstacle[2] + self.collision_distance/2.0:
                    # Agent is inside the obstacle, move it outside
                    direction = self.positions[agent] - obstacle[:2]
                    direction /= np.linalg.norm(direction)
                    self.positions[agent] = obstacle[:2] + direction * (obstacle[2] + self.collision_distance/2.0)

        self.distances = np.array([[self._periodic_distance(self.positions[agent1], self.positions[agent2]) for agent2 in self.agents] for agent1 in self.agents])

        # Check interactions and update rewards
        self._rewards = self._get_rewards(actions)

        # Termination and truncation: no specific conditions in your environment
        terminations = {agent: False for agent in self.agents}
        if self.timesteps_left <= 0:
            truncations = {agent: True for agent in self.agents}
        else:
            self.timesteps_left -= 1      
            truncations = {agent: False for agent in self.agents}

        # Get observations and infos
        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        return observations, self._rewards, terminations, truncations, infos

    def _periodic_distance(self, position1, position2):
            diff = position1 - position2
            # Apply periodic boundary by adjusting the difference to wrap around
            diff[0] -= np.round(diff[0] / (2 * self.x_limit)) * 2 * self.x_limit
            diff[1] -= np.round(diff[1] / (2 * self.y_limit)) * 2 * self.y_limit
            return np.linalg.norm(diff)

    def _get_obs(self, agent):
        agent_pos = self.positions[agent]
        agent_vel = self.velocities[agent]
        agent_angle = self.orientations[agent]
        
        return np.array([*agent_pos, *agent_vel, agent_angle, self.agent_types[agent]], dtype=np.float32)
    
    def _get_rewards(self, actions):
        rewards = {agent: 0.0 for agent in self.agents}

        for first_agent_index, first_agent in enumerate(self.agents):
            for second_agent_index, second_agent in enumerate(self.agents):
                if first_agent == second_agent:
                    continue
                distance = self.distances[first_agent_index, second_agent_index]

                first_agent_type = self.agent_types[first_agent]
                second_agent_type = self.agent_types[second_agent]
                
                # Predator has +1 type value compared to its prey
                if first_agent_type == second_agent_type + 1:
                    # first_agent is predator, second_agent is prey
                    if distance < self.kill_distance:
                        rewards[first_agent] += 100.0
                    else:
                        rewards[first_agent] -= 0.01 * distance # Penalize for being far from prey. Adjust/change if needed
                elif first_agent_type == second_agent_type - 1:
                    # first_agent is prey, second_agent is predator
                    if distance < self.kill_distance:
                        rewards[first_agent] -= 100.0
                    else:
                        rewards[first_agent] += 0.01 * distance # Reward for being far from predator. Adjust/change if needed
                else:
                    # They are not interacting
                    continue

        # Penalize for high actions
        for agent, action in actions.items():
            rotation_force, propulsion_force = action
            rewards[agent] -= 0.1 * np.abs(rotation_force)
            rewards[agent] -= 0.001 * np.abs(propulsion_force)
        
        return rewards

    def render(self):
        if not hasattr(self, 'fig'):
            self.fig, self.ax = plt.subplots(figsize=(6, 6))
            self.ax.set_xlim(-self.x_limit, self.x_limit)
            self.ax.set_ylim(-self.y_limit, self.y_limit)

            # Scatter plots for preys and predators
            self.prey_scatter = self.ax.scatter([], [], c='blue', label='Preys')
            self.predator_scatter = self.ax.scatter([], [], c='orange', label='Predators')
            self.super_predator_scatter = self.ax.scatter([], [], c='red', label='Super Predators')

            # Create a list to store orientation lines
            self.prey_lines = []
            self.predator_lines = []
            self.super_predator_lines = []

            for (x, y, radius) in self.obstacles:
                circle = patches.Circle((x, y), radius=radius, edgecolor='grey', facecolor='grey')
                self.ax.add_patch(circle)

        # Clear orientation lines
        for line in self.prey_lines:
            line.remove()
        for line in self.predator_lines:
            line.remove()
        for line in self.super_predator_lines:
            line.remove()

        self.prey_lines = []
        self.predator_lines = []
        self.super_predator_lines = []

        # Extract positions for all agents
        prey_positions = np.array([self.positions[agent] for agent in self.agents if self.agent_types[agent] == 0])
        predator_positions = np.array([self.positions[agent] for agent in self.agents if self.agent_types[agent] == 1])
        super_predator_positions = np.array([self.positions[agent] for agent in self.agents if self.agent_types[agent] == 2])

        prey_orientations = np.array([self.orientations[agent] for agent in self.agents if self.agent_types[agent] == 0])
        predator_orientations = np.array([self.orientations[agent] for agent in self.agents if self.agent_types[agent] == 1])
        super_predator_orientations = np.array([self.orientations[agent] for agent in self.agents if self.agent_types[agent] == 2])

        self.prey_scatter.set_offsets(prey_positions)
        self.predator_scatter.set_offsets(predator_positions)
        self.super_predator_scatter.set_offsets(super_predator_positions)

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
        
        # Draw orientation lines for super predators
        if len(super_predator_positions) > 0:
            for pos, orient in zip(super_predator_positions, super_predator_orientations):
                end_x = pos[0] + np.cos(orient) * 0.5  # Adjust the length as needed
                end_y = pos[1] + np.sin(orient) * 0.5
                line, = self.ax.plot([pos[0], end_x], [pos[1], end_y], c='red')
                self.super_predator_lines.append(line)

        plt.pause(0.01)

    def _get_observation_space(self):
        observation_space = spaces.Box(
            low=np.array([-self.x_limit, -self.y_limit, -np.inf, -np.inf, -np.pi, 0]), 
            high=np.array([self.x_limit, self.y_limit, np.inf, np.inf, np.pi, 2]), 
            dtype=np.float64)
        return observation_space
    
    def _get_action_space(self):
        action_space = spaces.Box(low=np.array([-1.0, 0.0]), high=np.array([1.0, 1.0]), dtype=np.float64)
        return action_space
    