import logging
from copy import deepcopy
import gymnasium as gym
import numpy as np
import time

from f110_gym.envs.base_classes import Simulator, Integrator

# rendering
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800


class F110Env(gym.Env):
    """OpenAI gym environment for F1TENTH.
    
    Env should be initialized by calling gym.make('f110_gym:f110-v1', **kwargs)

    Attributes:
        config (Config): Config object.
        map_names (list): List of map names to be used in the environment.
        render_mode (str): Rendering mode. Can be 'human', 'human_fast', or None.
        params (VehicleParams): Vehicle parameters.
        num_agents (int): Number of agents in the environment.
        timestep (float): Simulation timestep.
        ego_idx (int): Index of the ego vehicle.
        random_start_pose (bool): Whether to randomize the start pose of the ego vehicle.
        seed (int): Seed for the environment.
        start_thresh (float): Radius to consider done.
        integrator (Integrator): Integrator to use for the simulation.
        renderer (Renderer): Renderer object.
        current_obs (dict): Current observation.
        render_callbacks (list): List of render callbacks.
        sim (Simulator): Simulator object.
        map_names (str): Names of the maps to be used in the environment.
        map_ext (str): Map file extension.
        map_paths (dict): Dictionary of map paths.
        wpt_paths (dict): Dictionary of waypoint paths.
        current_map_name (str): Name of the current map.
        random_map (bool): Whether to use a random map.
        poses_x (list): List of x positions of the ego vehicle.
        poses_y (list): List of y positions of the ego vehicle.
        poses_theta (list): List of theta positions of the ego vehicle.
        collisions (np.ndarray): Array of collisions.
        control_to_sim_ratio (int): Ratio of control to simulation timesteps.
        near_start (bool): Whether the ego vehicle is near the start position.
        lap_times (list): List of lap times.
        num_toggles (int): Number of toggles.
        near_starts (np.ndarray): Array of whether the ego vehicle is near the start position.
        toggle_list (np.ndarray): Array of toggles.
        start_xs (np.ndarray): Array of x positions of the start position.
        start_ys (np.ndarray): Array of y positions of the start position.
        start_thetas (np.ndarray): Array of theta positions of the start position.
        start_rot (np.ndarray): Array of start rotations.
        render_obs (dict): Observation to be rendered.
        action_space (gym.spaces.Box): Action space.
        observation_space (gym.spaces.Dict): Observation space.
        metadata (dict): Metadata.
        """

    metadata = {'render_modes': ['human', 'human_fast']}

    def __init__(self, config, map_names, render_mode=None):
        self.config = config
        self._render_mode = render_mode

        self.params = self.config.sim.vehicle_params
        self.num_agents = 1
        self.timestep = self.config.sim.dt

        self.ego_idx = 0
        self.random_start_pose: bool = self.config.sim.random_start_pose
        self.seed = 12345
        self.start_thresh = 0.5  # Radius to consider done
        self.integrator = Integrator.RK4

        self.renderer = None
        self.current_obs = None
        self.render_callbacks = []

        # Initialize simulator
        self.sim = Simulator(self.params, self.num_agents, self.seed, time_step=self.timestep,
                             integrator=self.integrator)

        # Select and load the map data
        self.current_map_name = None
        if len(map_names) > 1:
            self.random_map = True
        elif len(map_names) == 1:
            self.random_map = False
        else:
            raise Exception('There must the something wrong with the maps.')

        self.map_names: list = map_names
        self.map_ext = self.config.maps.map_ext
        self.map_paths = dict()
        self.wpt_paths = dict()
        for map_name in self.map_names:
            self.map_paths[map_name] = self.config.maps.map_path + f'{map_name}/{map_name}_map'
            self.wpt_paths[map_name] = self.config.maps.map_path + f'{map_name}/{map_name}_raceline.csv'

        self._check_init_render()
        self._shuffle_map()

        # env states
        self.poses_x = []
        self.poses_y = []
        self.poses_theta = []
        self.collisions = np.zeros((self.num_agents,))

        self.control_to_sim_ratio = int(self.config.sim['controller_dt'] / self.config.sim['dt'])

        # loop completion
        self.near_start = True
        self.num_toggles = 0

        # race info
        self.lap_times = np.zeros((self.num_agents,))
        self.lap_counts = np.zeros((self.num_agents,), dtype=np.int8)
        self.current_time = 0.0

        # finish line info
        self.num_toggles = 0
        self.near_start = True
        self.near_starts = np.array([True] * self.num_agents)
        self.toggle_list = np.zeros((self.num_agents,))
        self.start_xs = np.zeros((self.num_agents,))
        self.start_ys = np.zeros((self.num_agents,))
        self.start_thetas = np.zeros((self.num_agents,))
        self.start_rot = np.eye(2)

        # stateful observations for rendering
        self.render_obs = None
        self.action_space = gym.spaces.Box(
            low=np.array([self.params['s_min'], self.params['v_min']]),
            high=np.array([self.params['s_max'], self.params['v_max']]),
            shape=(2,),
            dtype=np.float64
        )

        # Note: Dict is an ordered dict
        self.observation_space = gym.spaces.Dict(
            {
                'ego_idx': gym.spaces.Box(low=0, high=1, shape=(1,), dtype=int),
                'aaa_scans': gym.spaces.Box(low=0, high=31, shape=(1, 1080), dtype=np.float64),
                'poses_x': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64),
                'poses_y': gym.spaces.Box(low=--np.inf, high=-np.inf, shape=(1,), dtype=np.float64),
                'poses_theta': gym.spaces.Box(low=-2 * np.pi, high=2 * np.pi, shape=(1,), dtype=np.float64),
                'linear_vels_x': gym.spaces.Box(
                    low=self.params['v_min'], high=self.params['v_max'], shape=(1,), dtype=np.float64
                ),
                'linear_vels_y': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64),
                'ang_vels_z': gym.spaces.Box(low=-100, high=100, shape=(1,), dtype=np.float64),
                'collisions': gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.bool),
                'lap_times': gym.spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float64),
                'lap_counts': gym.spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.int8),
                'prev_action': gym.spaces.Box(
                    low=np.array([self.params['s_min'], self.params['v_min']]),
                    high=np.array([self.params['s_max'], self.params['v_max']]),
                    dtype=np.float64
                ),
                'slip_angle': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64),
                'yaw_rate': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64),
                'acc_x': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64),
                'acc_y': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64),
            }
        )

    @property
    def render_mode(self):
        """Get the rendering mode."""
        return self._render_mode

    @render_mode.setter
    def render_mode(self, render_mode):
        """Set the rendering mode."""
        self._render_mode = render_mode
        self._check_init_render()

    def _check_init_render(self):
        """Initialize the renderer if needed."""
        if self._render_mode in ['human', 'human_fast']:
            if self.renderer is None:
                # first call, initialize everything
                from f110_gym.envs.rendering import EnvRenderer
                from pyglet import options as pygl_options
                pygl_options['debug_gl'] = False
                self.renderer = EnvRenderer(WINDOW_W, WINDOW_H)
        else:
            if self.renderer is not None:
                self.renderer.close()
                self.renderer = None

    def _check_done(self):
        """ Check if the current rollout is done.

        Returns:
            done (bool): whether the rollout is done
            toggle_list (list[int]): each agent's toggle list for crossing the finish zone
        """

        # This is assuming 2 agents
        left_t = 2
        right_t = 2

        poses_x = np.array(self.poses_x) - self.start_xs
        poses_y = np.array(self.poses_y) - self.start_ys
        delta_pt = np.dot(self.start_rot, np.stack((poses_x, poses_y), axis=0))
        temp_y = delta_pt[1, :]
        idx1 = temp_y > left_t
        idx2 = temp_y < -right_t
        temp_y[idx1] -= left_t
        temp_y[idx2] = -right_t - temp_y[idx2]
        temp_y[np.invert(np.logical_or(idx1, idx2))] = 0

        dist2 = delta_pt[0, :] ** 2 + temp_y ** 2
        closes = dist2 <= 0.1
        for i in range(self.num_agents):
            if closes[i] and not self.near_starts[i]:
                self.near_starts[i] = True
                self.toggle_list[i] += 1
            elif not closes[i] and self.near_starts[i]:
                self.near_starts[i] = False
                self.toggle_list[i] += 1
            self.lap_counts[i] = self.toggle_list[i] // 2
            if self.toggle_list[i] <= 4:
                self.lap_times[i] = self.current_time

        done = (self.collisions[self.ego_idx]) or np.all(self.toggle_list >= 4)

        return bool(done), self.toggle_list >= 4

    def _update_state(self, obs_dict):
        """Update the env's states according to the observations.

        Args:
            obs_dict (dict): dictionary of observation
        """
        self.poses_x = obs_dict['poses_x']
        self.poses_y = obs_dict['poses_y']
        self.poses_theta = obs_dict['poses_theta']
        self.collisions = obs_dict['collisions']

    def step(self, action):
        """ Step function for the gym env.

        Args:
            action (np.ndarray(num_agents, 2))

        Returns:
            obs (dict): observation of the current step
            reward (float, default=self.timestep): step reward, currently is physics timestep
            done (bool): if the simulation is done
            info (dict): auxiliary information dictionary
        """

        # call simulation step
        if len(action.shape) < 2:
            action = np.expand_dims(action, 0)

        for _ in range(self.control_to_sim_ratio):
            # steer, vel
            obs = self.sim.step(action)
            self.current_time = self.current_time + self.timestep
            self.current_obs = obs
            # update data member
            self._update_state(obs)  # <- this here assigns the collision to the env
            # check done
            done, toggle_list = self._check_done()  # <- check the done of the env not the sim
            if done:
                break
        obs['prev_action'] = action[0]
        obs['lap_times'] = self.lap_times
        obs['lap_counts'] = self.lap_counts

        info = {'checkpoint_done': toggle_list}
        truncated = np.all(toggle_list)  # bool(self.toggle_list >= 4)

        self.obs_ = deepcopy(obs)

        if self.render_mode == "human" or self.render_mode == "human_fast":
            # Update the render obs before drawing a new frame
            self.render_obs = {
                'ego_idx': obs['ego_idx'],
                'poses_x': obs['poses_x'],
                'poses_y': obs['poses_y'],
                'poses_theta': obs['poses_theta'],
                'lap_times': obs['lap_times'],
                'lap_counts': obs['lap_counts']
            }
            self.render()

        logging.debug(f'\nslip_angle {obs["slip_angle"]} and yaw rate {obs["yaw_rate"]}')
        logging.debug(f'vel_x {obs["linear_vels_x"]}, vel_y {obs["linear_vels_y"]}')
        logging.debug(f'acc_x {obs["acc_x"]}, acc_y {obs["acc_y"]}')

        return obs, self.calc_reward(), done, truncated, info

    def _shuffle_map(self, map_name=None):
        """Shuffle the map."""
        if map_name is None:
            idx = np.random.randint(0, len(self.map_names)) if len(self.map_names) > 1 else 0
            self.current_map_name = self.map_names[idx]
        else:
            self.current_map_name = map_name
        logging.debug(f"new map: {self.current_map_name}")
        map_path = self.map_paths[self.current_map_name]
        wpt_path = self.wpt_paths[self.current_map_name]

        self.update_map(map_path, self.map_ext)

        self.waypoints = np.loadtxt(wpt_path, delimiter=self.config.maps.wpt_delim,
                                    skiprows=self.config.maps.wpt_rowskip)
        self.start_positions = self.waypoints[:,
                               (self.config.maps.wpt_xind, self.config.maps.wpt_yind, self.config.maps.wpt_thind)]
        self.pose_start = self.start_positions[0, ...]

    def reset(self, options=None, seed=None, map_name=None):
        """Reset the gym environment by given poses.

        Args:
            options:
            poses (np.ndarray (num_agents, 3)): poses to reset agents to

        Returns:
            obs (dict): observation of the current step
            reward (float, default=self.timestep): step reward, currently is physics timestep
            done (bool): if the simulation is done
            info (dict): auxiliary information dictionary
        """
        if self.random_map:
            map_name = options['map_name'] if options is not None and 'map_name' in options else None
            self._shuffle_map(map_name)
        # reset counters and data members
        if self.random_start_pose:
            idx = np.random.randint(0, len(self.start_positions))
            poses = self.start_positions[idx, np.newaxis]
        else:
            poses = self.pose_start

        self.current_time = 0.0
        self.collisions = np.zeros((self.num_agents,), dtype=bool)
        self.num_toggles = 0
        self.near_start = True
        self.near_starts = np.array([True] * self.num_agents)
        self.toggle_list = np.zeros((self.num_agents,))

        # states after reset
        self.start_xs = poses[:, 0]
        self.start_ys = poses[:, 1]
        self.start_thetas = poses[:, 2]
        self.start_rot = np.array([[np.cos(-self.start_thetas[self.ego_idx]),
                                    -np.sin(-self.start_thetas[self.ego_idx])],
                                   [np.sin(-self.start_thetas[self.ego_idx]),
                                    np.cos(-self.start_thetas[self.ego_idx])]])

        # call reset to simulator
        self.sim.reset(poses)

        # get no input observations
        action = np.zeros((self.num_agents, 2))
        obs, reward, done, _, info = self.step(action)

        # Important for correct calculation: set time again to 0, has been increased by stepping the env once
        self.current_time = 0.0
        obs['lap_times'] = np.array([0.0])
        obs['prev_action'] = action[0]

        self.render_obs = {
            'ego_idx': obs['ego_idx'],
            'poses_x': obs['poses_x'],
            'poses_y': obs['poses_y'],
            'poses_theta': obs['poses_theta'],
            'lap_times': obs['lap_times'],
            'lap_counts': obs['lap_counts'],
        }

        if self.render_mode == "human":
            self.render()

        return obs, info

    def calc_reward(self):
        config = self.config.sim.reward
        step = 1 * config.step
        collision = self.current_obs['collisions'][0] * config.collision
        long_vel = self.current_obs['linear_vels_x'][0] * config.long_vel
        lat_vel = np.square(self.current_obs['linear_vels_y'][0]) * config.lat_vel
        # lat_vel = lat_vel if lat_vel < np.abs(long_vel) else np.abs(long_vel)
        reward = (step + collision + long_vel + lat_vel) * config.scaling
        logging.debug(f'step={step}, collision={collision}, long_vel={long_vel}, lat_vel={lat_vel}, reward={reward}')
        return reward

    def update_map(self, map_path, map_ext):
        """Updates the map used by simulation

        Args:
            map_path (str): absolute path to the map yaml file
            map_ext (str): extension of the map image file

        Returns:
            None
        """
        self.sim.set_map(map_path + '.yaml', map_ext)
        if self.renderer is not None:
            self.renderer.update_map(map_path, map_ext)

    def update_params(self, params, index=-1):
        """
        Updates the parameters used by simulation for vehicles

        Args:
            params (dict): dictionary of parameters
            index (int, default=-1): if >= 0 then only update a specific agent's params

        Returns:
            None
        """
        self.sim.update_params(params, agent_idx=index)

    def add_render_callback(self, callback_func):
        """
        Add extra drawing function to call during rendering.

        Args:
            callback_func (function (EnvRenderer) -> None): custom function to called during render()
        """

        self.render_callbacks.append(callback_func)

    def render(self):
        """Render the environment."""
        if self.render_mode is not None:

            self.renderer.update_obs(self.render_obs)

            for render_callback in self.render_callbacks:
                render_callback(self.renderer)

            self.renderer.dispatch_events()
            self.renderer.on_draw()
            self.renderer.flip()
            if self._render_mode == 'human':
                time.sleep(0.005)
            elif self._render_mode == 'human_fast':
                pass

    def close(self):
        """Close the environment."""
        if self.renderer is not None:
            self.renderer.close()
        self.renderer = None
        super().close()
