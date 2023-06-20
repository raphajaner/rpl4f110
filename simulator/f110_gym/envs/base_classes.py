import logging
from enum import Enum
import numpy as np

from f110_gym.envs.dynamic_models import vehicle_dynamics_st, pid
from f110_gym.envs.laser_models import ScanSimulator2D, check_ttc_jit, ray_cast
from f110_gym.envs.collision_models import get_vertices, collision_multiple
from numba import njit, jit


class Integrator(Enum):
    RK4 = 1
    Euler = 2


class RaceCar(object):
    """ Base level race car class, handles the physics and laser scan of a single vehicle.

    Attributes: params (dict): vehicle parameters dictionary is_ego (bool): ego identifier time_step (float): physics
    timestep num_beams (int): number of beams in laser fov (float): field of view of laser state (np.ndarray (7,
    )): state vector [x, y, theta, vel, steer_angle, ang_vel, slip_angle] odom (np.ndarray(13, )): odometry vector [
    x, y, z, qx, qy, qz, qw, linear_x, linear_y, linear_z, angular_x, angular_y, angular_z] accel (float): current
    acceleration input steer_angle_vel (float): current steering velocity input in_collision (bool): collision indicator

    """

    def __init__(self, params, seed, is_ego=False, time_step=0.01, num_beams=1080, fov=4.7,
                 integrator=Integrator.Euler):
        """ Init function.

        Args:
            params (dict): vehicle parameter dictionary, includes
                {'mu', 'C_Sf', 'C_Sr', 'lf', 'lr', 'h', 'm', 'I', 's_min', 's_max',
                'sv_min', 'sv_max', 'v_switch', 'a_max': 9.51, 'v_min', 'v_max', 'length', 'width'}
            is_ego (bool, default=False): ego identifier
            time_step (float, default=0.01): physics sim time step
            num_beams (int, default=1080): number of beams in the laser scan
            fov (float, default=4.7): field of view of the laser
        """

        self.scan_simulator = None
        self.cosines = None
        self.scan_angles = None
        self.side_distances = None

        # initialization
        self.params = params
        # Accessing params from dict is expensive, so we cache them here
        self.mu = params['mu']
        self.C_Sf = params['C_Sf']
        self.C_Sr = params['C_Sr']
        self.lf = params['lf']
        self.lr = params['lr']
        self.h = params['h']
        self.m = params['m']
        self.I = params['I']
        self.s_min = params['s_min']
        self.s_max = params['s_max']
        self.sv_min = params['sv_min']
        self.sv_max = params['sv_max']
        self.v_switch = params['v_switch']
        self.a_max = params['a_max']
        self.v_min = params['v_min']
        self.v_max = params['v_max']
        self.length = params['length']
        self.width = params['width']

        self.seed = seed
        self.is_ego = is_ego
        self.time_step = time_step
        self.num_beams = num_beams
        self.fov = fov
        self.integrator = integrator
        # state is [x, y, steer_angle, vel, yaw_angle, yaw_rate, slip_angle]
        self.state = np.zeros((7,))
        # Add another state: Lateral acceleration (no independent state)
        self.a_x = np.zeros((1,))
        self.a_y = np.zeros((1,))

        # pose of opponents in the world
        self.opp_poses = None

        # control inputs
        self.accel = 0.0
        self.steer_angle_vel = 0.0

        # steering delay buffer
        self.steer_buffer = np.empty((0,))
        self.steer_buffer_size = 2

        # collision identifier
        self.in_collision = False

        # collision threshold for iTTC to environment
        self.ttc_thresh = 0.005

        # initialize scan sim
        if self.scan_simulator is None:
            self.scan_rng = np.random.default_rng(seed=self.seed)
            self.scan_simulator = ScanSimulator2D(num_beams, fov)

            scan_ang_incr = self.scan_simulator.get_increment()

            # angles of each scan beam, distance from lidar to edge of car at each beam, and precomputed cosines of each angle
            self.cosines = np.zeros((num_beams,))
            self.scan_angles = np.zeros((num_beams,))
            self.side_distances = np.zeros((num_beams,))

            dist_sides = self.width / 2.
            dist_fr = (self.lf + self.lr) / 2.

            for i in range(num_beams):
                angle = -fov / 2. + i * scan_ang_incr
                self.scan_angles[i] = angle
                self.cosines[i] = np.cos(angle)

                if angle > 0:
                    if angle < np.pi / 2:
                        # between 0 and pi/2
                        to_side = dist_sides / np.sin(angle)
                        to_fr = dist_fr / np.cos(angle)
                        self.side_distances[i] = min(to_side, to_fr)
                    else:
                        # between pi/2 and pi
                        to_side = dist_sides / np.cos(angle - np.pi / 2.)
                        to_fr = dist_fr / np.sin(angle - np.pi / 2.)
                        self.side_distances[i] = min(to_side, to_fr)
                else:
                    if angle > -np.pi / 2:
                        # between 0 and -pi/2
                        to_side = dist_sides / np.sin(-angle)
                        to_fr = dist_fr / np.cos(-angle)
                        self.side_distances[i] = min(to_side, to_fr)
                    else:
                        # between -pi/2 and -pi
                        to_side = dist_sides / np.cos(-angle - np.pi / 2)
                        to_fr = dist_fr / np.sin(-angle - np.pi / 2)
                        self.side_distances[i] = min(to_side, to_fr)

    def update_params(self, params):
        """ Updates the physical parameters of the vehicle with params."""
        self.params = params
        self.mu = params['mu']
        self.C_Sf = params['C_Sf']
        self.C_Sr = params['C_Sr']
        self.lf = params['lf']
        self.lr = params['lr']
        self.h = params['h']
        self.m = params['m']
        self.I = params['I']
        self.s_min = params['s_min']
        self.s_max = params['s_max']
        self.sv_min = params['sv_min']
        self.sv_max = params['sv_max']
        self.v_switch = params['v_switch']
        self.a_max = params['a_max']
        self.v_min = params['v_min']
        self.v_max = params['v_max']
        self.length = params['length']
        self.width = params['width']

    def set_map(self, map_path, map_ext):
        """ Sets the map for scan simulator.
        Args:
            map_path (str): absolute path to the map yaml file
            map_ext (str): extension of the map image file
        """
        self.scan_simulator.set_map(map_path, map_ext)

    def reset(self, pose):
        """ Resets the vehicle to a pose.

        Args:
            pose (np.ndarray (3, )): pose to reset the vehicle to
        """
        # clear control inputs
        self.accel = 0.0
        self.steer_angle_vel = 0.0
        # clear collision indicator
        self.in_collision = False
        # clear state
        self.state = np.zeros((7,))
        self.state[0:2] = pose[0:2]
        self.state[4] = pose[2]
        self.steer_buffer = np.empty((0,))
        # reset scan random generator
        self.scan_rng = np.random.default_rng(seed=self.seed)
        self.a_lat = np.zeros((1,))

    def ray_cast_agents(self, scan):
        """ Ray cast onto other agents in the env, modify original scan.

        Args:
            scan (np.ndarray, (n, )): original scan range array
        Returns:
            new_scan (np.ndarray, (n, )): modified scan
        """

        # starting from original scan
        new_scan = scan
        # loop over all opponent vehicle poses
        for opp_pose in self.opp_poses:
            # get vertices of current oppoenent
            opp_vertices = get_vertices(opp_pose, self.length, self.width)

            new_scan = ray_cast(np.append(self.state[0:2], self.state[4]), new_scan, self.scan_angles, opp_vertices)

        return new_scan

    def check_ttc(self, current_scan):
        """ Check iTTC against the environment.

        Sets vehicle states accordingly if collision occurs. Note that this does NOT check collision with other
        agents. State is [x, y, steer_angle, vel, yaw_angle, yaw_rate, slip_angle]
        Args:
            current_scan(np.ndarray, (n, )): current scan range array
        """

        in_collision = check_ttc_jit(current_scan,
                                     self.state[3],
                                     self.scan_angles,
                                     self.cosines,
                                     self.side_distances,
                                     self.ttc_thresh)

        # if in collision stop vehicle
        if in_collision:
            self.state[3:] = 0.
            self.accel = 0.0
            self.steer_angle_vel = 0.0

        # update state
        self.in_collision = in_collision

        return in_collision

    def update_pose(self, raw_steer, vel):
        """ Steps the vehicle's physical simulation.

        Args:
            steer (float): desired steering angle
            vel (float): desired longitudinal velocity

        Returns:
            current_scan
        """

        # state is [x, y, steer_angle, vel, yaw_angle, yaw_rate, slip_angle]

        # steering delay
        # steer = 0.0
        if self.steer_buffer.shape[0] < self.steer_buffer_size:
            steer = 0.0
            self.steer_buffer = np.append(raw_steer, self.steer_buffer)
        else:
            steer = self.steer_buffer[-1]
            self.steer_buffer = self.steer_buffer[:-1]
            self.steer_buffer = np.append(raw_steer, self.steer_buffer)

        # steering angle velocity input to steering velocity acceleration input
        accl, sv = pid(vel, steer, self.state[3], self.state[2], self.sv_max, self.a_max,
                       self.v_max, self.v_min)

        self.a_x = accl

        if self.integrator is Integrator.RK4:
            # RK4 integration
            k1 = self._get_vehicle_dynamics_st(self.state, sv, accl)

            k2_state = self.state + self.time_step * (k1 / 2)
            k2 = self._get_vehicle_dynamics_st(k2_state, sv, accl)

            k3_state = self.state + self.time_step * (k2 / 2)
            k3 = self._get_vehicle_dynamics_st(k3_state, sv, accl)

            k4_state = self.state + self.time_step * k3
            k4 = self._get_vehicle_dynamics_st(k4_state, sv, accl)

            # dynamics integration
            f = (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            self.state = self.state + self.time_step * f
            # Note: f is the state's derivative, i.e., x'
            # Later acceleration is the velocity multiplied with the difference of the yaw and slip rate (= velocity)
            self.a_y = self.state[3] * (self.state[5] - f[6])
            # print(f'v {self.state[3]} : yaw rate {self.state[5]} : slip rate {f[6]}')

        elif self.integrator is Integrator.Euler:
            f = self._get_vehicle_dynamics_st(self.state, sv, accl)
            self.state = self.state + self.time_step * f
            self.a_y = self.state[3] * (self.state[5] - f[6])

        # else:
        #     raise SyntaxError(
        #         f"Invalid Integrator Specified. Provided {self.integrator.name}. Please choose RK4 or Euler")

        # bound yaw angle
        if self.state[4] > 2 * np.pi:
            self.state[4] = self.state[4] - 2 * np.pi
        elif self.state[4] < 0:
            self.state[4] = self.state[4] + 2 * np.pi

        # update scan
        current_scan = self.scan_simulator.scan(np.append(self.state[0:2], self.state[4]), self.scan_rng)

        return current_scan

    def _get_vehicle_dynamics_st(self, state, sv, accl):
        f = vehicle_dynamics_st(
            state,
            np.array([sv, accl]),
            self.mu,
            self.C_Sf,
            self.C_Sr,
            self.lf,
            self.lr,
            self.h,
            self.m,
            self.I,
            self.s_min,
            self.s_max,
            self.sv_min,
            self.sv_max,
            self.v_switch,
            self.a_max,
            self.v_min,
            self.v_max
        )
        return f

    def update_opp_poses(self, opp_poses):
        """ Updates the vehicle's information on other vehicles.

        Args:
            opp_poses (np.ndarray(num_other_agents, 3)): updated poses of other agents
        """
        self.opp_poses = opp_poses

    def update_scan(self, agent_scans, agent_index):
        """ Steps the vehicle's laser scan simulation.

        Separated from update_pose because needs to update scan based on NEW poses of agents in the environment

        Args:
            agent scans list (modified in-place),
            agent index (int)
        """

        current_scan = agent_scans[agent_index]

        # check ttc
        self.check_ttc(current_scan)

        # ray cast other agents to modify scan
        new_scan = self.ray_cast_agents(current_scan)

        agent_scans[agent_index] = new_scan


class Simulator(object):
    """Simulator class, handles the interaction and update of all vehicles in the environment.

    Attributes:
        num_agents (int): number of agents in the environment
        time_step (float): physics time step
        agent_poses (np.ndarray(num_agents, 3)): all poses of all agents
        agents (list[RaceCar]): container for RaceCar objects
        collisions (np.ndarray(num_agents, )): array of collision indicator for each agent
        collision_idx (np.ndarray(num_agents, )): which agent is each agent in collision with

    """

    def __init__(self, params, num_agents, seed, time_step=0.01, ego_idx=0, integrator=Integrator.RK4):
        """Init function.

        Args: params (dict): vehicle parameter dictionary, includes {'mu', 'C_Sf', 'C_Sr', 'lf', 'lr', 'h', 'm', 'I',
        's_min', 's_max', 'sv_min', 'sv_max', 'v_switch', 'a_max', 'v_min', 'v_max', 'length', 'width'} num_agents (
        int): number of agents in the environment seed (int): seed of the rng in scan simulation time_step (float,
        default=0.01): physics time step ego_idx (int, default=0): ego vehicle's index in list of agents
        """
        self.num_agents = num_agents
        self.seed = seed
        self.time_step = time_step
        self.ego_idx = ego_idx
        self.params = params
        self.length = params['length']
        self.width = params['width']
        self.agent_poses = np.empty((self.num_agents, 3))
        self.agents = []
        self.collisions = np.zeros((self.num_agents,))
        self.collision_idx = -1 * np.ones((self.num_agents,))

        # initializing agents
        for i in range(self.num_agents):
            if i == ego_idx:
                ego_car = RaceCar(params, self.seed, is_ego=True, time_step=self.time_step, integrator=integrator)
                self.agents.append(ego_car)
            else:
                agent = RaceCar(params, self.seed, is_ego=False, time_step=self.time_step, integrator=integrator)
                self.agents.append(agent)

    def set_map(self, map_path, map_ext):
        """Sets the map of the environment and sets the map for scan simulator of each agent.

        Args:
            map_path (str): path to the map yaml file
            map_ext (str): extension for the map image file
        """
        for agent in self.agents:
            agent.set_map(map_path, map_ext)

    def update_params(self, params, agent_idx=-1):
        """Updates the params of agents, if an index of an agent is given, update only that agent's params

        Args:
            params (dict): dictionary of params, see details in docstring of __init__
            agent_idx (int, default=-1): index for agent that needs param update, if negative, update all agents
        """
        if agent_idx < 0:
            # update params for all
            for agent in self.agents:
                agent.update_params(params)
        elif 0 <= agent_idx < self.num_agents:
            # only update one agent's params
            self.agents[agent_idx].update_params(params)
        else:
            # index out of bounds, throw error
            raise IndexError('Index given is out of bounds for list of agents.')

    def check_collision(self):
        """Checks for collision between agents using GJK and agents' body vertices."""
        all_vertices = np.empty((self.num_agents, 4, 2))  # get vertices of all agents
        for i in range(self.num_agents):
            all_vertices[i, :, :] = get_vertices(np.append(self.agents[i].state[0:2], self.agents[i].state[4]),
                                                 self.length, self.width)
        self.collisions, self.collision_idx = collision_multiple(all_vertices)

    def step(self, control_inputs):
        """Steps the simulation environment.

        Args:
            control_inputs (np.ndarray (num_agents, 2)): control inputs of all agents,
            first column is desired steering angle, second column is desired velocity
        
        Returns:
            observations (dict): dictionary for observations: poses of agents,
            current laser scan of each agent, collision indicators, etc.
        """

        agent_scans = []

        # looping over agents
        for i, agent in enumerate(self.agents):
            # update each agent's pose
            current_scan = agent.update_pose(control_inputs[i, 0], control_inputs[i, 1])
            agent_scans.append(current_scan)
            # update sim's information of agent poses
            self.agent_poses[i, :] = np.append(agent.state[0:2], agent.state[4])

        # check collisions between all agents
        self.check_collision()

        for i, agent in enumerate(self.agents):
            # update agent's information on other agents
            opp_poses = np.concatenate((self.agent_poses[0:i, :], self.agent_poses[i + 1:, :]), axis=0)
            agent.update_opp_poses(opp_poses)
            # update each agent's current scan based on other agents
            agent.update_scan(agent_scans, i)
            # update agent collision with environment
            if agent.in_collision:
                self.collisions[i] = 1.

        # fill in observations
        # state is [x, y, steer_angle, vel, yaw_angle, yaw_rate, slip_angle]
        # collision_angles is removed from observations
        agent = self.agents[0]
        observations = {
            'ego_idx': np.array([self.ego_idx]),
            'aaa_scans': np.array(agent_scans[0])[np.newaxis, ...],  # Ensure that first in an ordered dict
            'poses_x': np.array(agent.state[0])[np.newaxis, ...],
            'poses_y': np.array(agent.state[1])[np.newaxis, ...],
            'poses_theta': np.array(agent.state[4])[np.newaxis, ...],
            'linear_vels_x': np.array(np.cos(agent.state[6]) * agent.state[3])[np.newaxis, ...],
            'linear_vels_y': np.array(np.sin(agent.state[6]) * agent.state[3]
                                      if agent.state.shape[-1] == 7
                                      else 0.0)[np.newaxis, ...],
            'ang_vels_z': np.array(agent.state[5])[np.newaxis, ...],
            'collisions': np.array([bool(self.collisions)]),
            'slip_angle': np.array(agent.state[6])[np.newaxis, ...],
            'yaw_rate': np.array(agent.state[5])[np.newaxis, ...],
            'acc_x': np.array(agent.a_x)[np.newaxis, ...],
            'acc_y': np.array(agent.a_y)[np.newaxis, ...]
        }
        return observations

    def reset(self, poses):
        """Resets the simulation environment by given poses."""

        if poses.shape[0] != self.num_agents:
            raise ValueError('Number of poses for reset does not match number of agents.')

        # loop over poses to reset
        for i in range(self.num_agents):
            self.agents[i].reset(poses[i, :])
