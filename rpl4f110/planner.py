import logging
from abc import ABC
import torch
from copy import deepcopy
import gymnasium as gym
import numpy as np
from numba import njit


def render_callback(env_renderer, planner):
    """Callback function for rendering the environment by updating the camera to follow the car.
    Args:
        env_renderer (f110_gym.envs.env_renderer.EnvRenderer): Environment renderer.
        planner (planner.Planner): Planner.
    """
    x = env_renderer.cars[0].vertices[::2]
    y = env_renderer.cars[0].vertices[1::2]
    top, bottom, left, right = max(y), min(y), min(x), max(x)
    env_renderer.score_label.x = left
    env_renderer.score_label.y = top - 700
    env_renderer.left = left - 800
    env_renderer.right = right + 800
    env_renderer.top = top + 800
    env_renderer.bottom = bottom - 800
    planner.render_waypoints(env_renderer)


@njit(fastmath=False, cache=True)
def nearest_point_on_trajectory(point, trajectory):
    """Return the nearest point along the given piecewise linear trajectory.

    Note: Trajectories must be unique. If they are not unique, a divide by 0 error will occur

    Args:
        point(np.ndarray): size 2 numpy array
        trajectory: Nx2 matrix of (x,y) trajectory waypoints

    Returns:
        projection(np.ndarray): size 2 numpy array of the nearest point on the trajectory
        dist(float): distance from the point to the projection
        t(float): the t value of the projection along the trajectory
        min_dist_segment(int): the index of the segment of the trajectory that the projection is on
    """
    diffs = trajectory[1:, :] - trajectory[:-1, :]
    l2s = diffs[:, 0] ** 2 + diffs[:, 1] ** 2
    # this is equivalent to the elementwise dot product
    # dots = np.sum((point - trajectory[:-1,:]) * diffs[:,:], axis=1)
    dots = np.empty((trajectory.shape[0] - 1,))
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
    t = dots / l2s
    t[t < 0.0] = 0.0
    t[t > 1.0] = 1.0
    # t = np.clip(dots / l2s, 0.0, 1.0)
    projections = trajectory[:-1, :] + (t * diffs.T).T
    # dists = np.linalg.norm(point - projections, axis=1)
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp * temp))
    min_dist_segment = np.argmin(dists)
    return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], min_dist_segment


@njit(fastmath=False, cache=True)
def first_point_on_trajectory_intersecting_circle(point, radius, trajectory, t=0.0, wrap=False):
    """Return the first point along the given piecewise linear trajectory that intersects the given circle.

    Assumes that the first segment passes within a single radius of the point
    http://codereview.stackexchange.com/questions/86421/line-segment-to-circle-collision-algorithm

    Args:
        point(np.ndarray): size 2 numpy array
        radius(float): radius of the circle
        trajectory: Nx2 matrix of (x,y) trajectory waypoints
        t(float): the t value of the trajectory to start searching from
        wrap(bool): if True, wrap the trajectory around to the beginning if the end is reached

    Returns:
        projection(np.ndarray): size 2 numpy array of the nearest point on the trajectory
        dist(float): distance from the point to the projection
        t(float): the t value of the projection along the trajectory
        min_dist_segment(int): the index of the segment of the trajectory that the projection is on
    """
    start_i = int(t)
    start_t = t % 1.0
    first_t = None
    first_i = None
    first_p = None
    trajectory = np.ascontiguousarray(trajectory)
    for i in range(start_i, trajectory.shape[0] - 1):
        start = trajectory[i, :]
        end = trajectory[i + 1, :] + 1e-6
        V = np.ascontiguousarray(end - start)

        a = np.dot(V, V)
        b = 2.0 * np.dot(V, start - point)
        c = np.dot(start, start) + np.dot(point, point) - 2.0 * np.dot(start, point) - radius * radius
        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            continue
        #   print "NO INTERSECTION"
        # else:
        # if discriminant >= 0.0:
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2.0 * a)
        t2 = (-b + discriminant) / (2.0 * a)
        if i == start_i:
            if t1 >= 0.0 and t1 <= 1.0 and t1 >= start_t:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            if t2 >= 0.0 and t2 <= 1.0 and t2 >= start_t:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break
        elif t1 >= 0.0 and t1 <= 1.0:
            first_t = t1
            first_i = i
            first_p = start + t1 * V
            break
        elif t2 >= 0.0 and t2 <= 1.0:
            first_t = t2
            first_i = i
            first_p = start + t2 * V
            break
    # wrap around to the beginning of the trajectory if no intersection is found1
    if wrap and first_p is None:
        for i in range(-1, start_i):
            start = trajectory[i % trajectory.shape[0], :]
            end = trajectory[(i + 1) % trajectory.shape[0], :] + 1e-6
            V = end - start

            a = np.dot(V, V)
            b = 2.0 * np.dot(V, start - point)
            c = np.dot(start, start) + np.dot(point, point) - 2.0 * np.dot(start, point) - radius * radius
            discriminant = b * b - 4 * a * c

            if discriminant < 0:
                continue
            discriminant = np.sqrt(discriminant)
            t1 = (-b - discriminant) / (2.0 * a)
            t2 = (-b + discriminant) / (2.0 * a)
            if t1 >= 0.0 and t1 <= 1.0:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            elif t2 >= 0.0 and t2 <= 1.0:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break

    return first_p, first_i, first_t


@njit(fastmath=False, cache=True)
def get_actuation(pose_theta, lookahead_point, position, lookahead_distance, wheelbase):
    """Get the actuation for the given pose and lookahead point.

    Args:
        pose_theta(float): the current pose angle
        lookahead_point(np.ndarray): the lookahead point
        position(np.ndarray): the current position
        lookahead_distance(float): the lookahead distance
        wheelbase(float): the wheelbase of the vehicle

    Returns:
        speed(float): the speed to drive at
        steering_angle(float): the steering angle to drive at
    """
    waypoint_y = np.dot(np.array([np.sin(-pose_theta), np.cos(-pose_theta)]), lookahead_point[0:2] - position)
    speed = lookahead_point[2]
    if np.abs(waypoint_y) < 1e-6:
        return speed, 0.
    radius = 1 / (2.0 * waypoint_y / lookahead_distance ** 2)
    steering_angle = np.arctan(wheelbase / radius)
    return speed, steering_angle


class Planner:
    """Base class for planners."""

    def plan(self, pose_x, pose_y, pose_theta, lookahead_distance, vgain, n_next_points):
        """Plan a trajectory. Returns a list of (x, y, v) tuples."""
        raise NotImplementedError()


class PurePursuitPlanner(Planner):
    """Planner that uses pure pursuit to follow a trajectory."""

    def __init__(self, conf, wb, waypoints):
        self.wheelbase = wb
        self.conf = conf
        self.waypoints = waypoints
        self.max_reacquire = 20.
        self.drawn_waypoints = []

    def render_waypoints(self, e):
        """Render the waypoints e."""
        points = np.vstack((self.waypoints[:, self.conf.maps.wpt_xind], self.waypoints[:, self.conf.maps.wpt_yind])).T

        scaled_points = 50. * points

        for i in range(points.shape[0]):
            if len(self.drawn_waypoints) < points.shape[0]:
                from pyglet.gl import GL_POINTS
                b = e.batch.add(1, GL_POINTS, None, ('v3f/stream', [scaled_points[i, 0], scaled_points[i, 1], 0.]),
                                ('c3B/stream', [183, 193, 222]))
                self.drawn_waypoints.append(b)
            else:
                self.drawn_waypoints[i].vertices = [scaled_points[i, 0], scaled_points[i, 1], 0.]

    def _get_current_waypoint(self, waypoints, lookahead_distance, position, theta, n_next_points=5):
        """Get the current waypoint.

        Args:
            waypoints(np.ndarray): the waypoints
            lookahead_distance(float): the lookahead distance
            position(np.ndarray): the current position
            theta(float): the current pose angle
            n_next_points(int): the number of next points to return

        Returns:
            current_waypoint(np.ndarray): the current waypoint
        """
        wpts = np.vstack((self.waypoints[:, self.conf.maps.wpt_xind], self.waypoints[:, self.conf.maps.wpt_yind])).T
        nearest_point, nearest_dist, t, i = nearest_point_on_trajectory(position, wpts)
        if nearest_dist < lookahead_distance:
            lookahead_point, i2, t2 = first_point_on_trajectory_intersecting_circle(
                position, lookahead_distance, wpts,
                i + t, wrap=True
            )
            if i2 is None:
                return None

            current_waypoint = np.empty((n_next_points, 3))

            if i2 + n_next_points > wpts.shape[0]:
                o = i2 + n_next_points - wpts.shape[0]

                # x, y
                current_waypoint[:n_next_points - o, 0:2] = wpts[i2:, :]
                if o != 0:
                    current_waypoint[n_next_points - o:, 0:2] = wpts[:o, :]
                # speed
                current_waypoint[:n_next_points - o, 2] = waypoints[i2:, self.conf.maps.wpt_vind]
                if o != 0:
                    current_waypoint[n_next_points - o:, 2] = waypoints[:o, self.conf.maps.wpt_vind]
            else:
                # x, y
                current_waypoint[:, 0:2] = wpts[i2:i2 + n_next_points, :]
                # speed
                current_waypoint[:, 2] = waypoints[i2:i2 + n_next_points, self.conf.maps.wpt_vind]
            return current_waypoint

        elif nearest_dist < self.max_reacquire:
            current_waypoint = np.empty((n_next_points, 3))
            if i + n_next_points > wpts.shape[0]:
                o = i + n_next_points - wpts.shape[0]
                # x, y
                current_waypoint[:n_next_points - o, 0:2] = wpts[i:, :]
                if o != 0:
                    current_waypoint[n_next_points - o:, 0:2] = wpts[:o, :]
                # speed
                current_waypoint[:n_next_points - o, 2] = waypoints[i:, self.conf.maps.wpt_vind]
                if o != 0:
                    current_waypoint[n_next_points - o:, 2] = waypoints[:o, self.conf.maps.wpt_vind]
            else:
                # x, y
                current_waypoint[:, 0:2] = wpts[i:i + n_next_points, :]
                # speed
                current_waypoint[:, 2] = waypoints[i:i + n_next_points, self.conf.maps.wpt_vind]
            return current_waypoint
        else:
            return None

    def plan(self, pose_x, pose_y, pose_theta, lookahead_distance, vgain, n_next_points):
        """Plan a trajectoryt to follow the waypoints.

        Args:
            pose_x(float): the current x position
            pose_y(float): the current y position
            pose_theta(float): the current pose angle
            lookahead_distance(float): the lookahead distance
            vgain(float): the speed gain
            n_next_points(int): the number of next points to return

        Returns:
            speed(float): the speed
            steering_angle(float): the steering angle
            lookahead_point(np.ndarray): the lookahead point
        """
        position = np.array([pose_x, pose_y])
        lookahead_point = self._get_current_waypoint(
            self.waypoints,
            lookahead_distance,
            position,
            pose_theta,
            n_next_points=n_next_points
        )

        if lookahead_point is None:
            return np.zeros((n_next_points, 3))  # TODO: Will/should throw an error since lookahead point is missing

        logging.debug(f'lookahead_point dtype {lookahead_point.dtype}, shape {lookahead_point.shape}')

        speed, steering_angle = get_actuation(
            pose_theta, lookahead_point[0], position, lookahead_distance,
            self.wheelbase
        )
        speed = vgain * speed

        return speed, steering_angle, lookahead_point


class GymActionObservationWrapper(gym.Wrapper):
    """A wrapper that modifies the action and observation space of the environment."""

    def __init__(self, env):
        super().__init__(env)

    def observation(self, observation):
        """Returns a modified observation."""
        raise NotImplementedError

    def action(self, action):
        """Returns a modified action before :meth:`env.step` is called."""

        raise NotImplementedError

    def reverse_action(self, action):
        """Returns a reversed ``action``."""
        raise NotImplementedError

    def step(self, action):
        """Returns a modified observation using :meth:`self.observation` after calling :meth:`env.step`."""
        observation, reward, terminated, truncated, info = self.env.step(self.action(action))
        observation_ = self.observation(observation)
        info['obs_'] = deepcopy(observation_)
        return observation_, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Resets the environment, returning a modified observation using :meth:`self.observation`."""
        observation, info = self.env.reset(**kwargs)
        observation_ = self.observation(observation)
        info['obs_'] = deepcopy(observation_)
        return observation_, info


class PlannerEnvWrapper(GymActionObservationWrapper, ABC):
    """A wrapper that adds a planner to the environment.

    Attributes:
        planner (PurePursuitPlanner): the planner
        planner_work (dict): the planner's work
        action_scaling (np.ndarray): the action scaling
        n_next_points (int): the number of next points to return
        skip_next_points (int): the number of next points to skip
        observation_space (gym.spaces.Box): the observation space
        action_applied_space (gym.spaces.Box): the action applied space
        planner_action (nd.array): the planner's action
    """

    def __init__(self,
                 env,
                 config,
                 ):
        super().__init__(env)

        self.planner = PurePursuitPlanner(
            config,
            wb=(config.sim.vehicle_params.lf + config.sim.vehicle_params.lr),
            waypoints=self.waypoints
        )

        self.planner_work = {
            'tlad': config.planner.tlad,
            'vgain': config.planner.vgain
        }
        self.action_scaling = np.array(config.sim.action_scaling)
        self.n_next_points = config.planner.n_next_points
        self.skip_next_points = config.planner.skip_next_points

        # Extend the base env's obs space by the planner's action
        # Note: Important to deepcopy, otherwise the super().observation_space will be altered
        self.observation_space = deepcopy(self.observation_space)

        self.observation_space['action_planner'] = gym.spaces.Box(
            low=np.array([self.params['s_min'], self.params['v_min']]),
            high=np.array([self.params['s_max'], self.params['v_max']]),
            dtype=np.float64)

        self.observation_space['lookahead_points_relative'] = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(int(self.n_next_points * 3 / self.skip_next_points),),
            dtype=np.float64
        )
        self.action_applied_space = deepcopy(self.env.action_space)

        # Original action space of self.env stays untouched
        self.action_space = gym.spaces.Box(
            low=np.array([-np.inf, -np.inf]),
            high=np.array([np.inf, np.inf]),
            shape=(2,),
            dtype=np.float64
        )

        self.planner_action = None

    @property
    def waypoints(self):
        """Returns the waypoints."""
        return self.env.waypoints

    def observation(self, obs):
        """Returns a modified observation."""
        self.planner.waypoints = self.env.waypoints
        speed, steering_angle, lookahead_point_global = self.planner.plan(
            obs['poses_x'][0],
            obs['poses_y'][0],
            obs['poses_theta'][0],
            self.planner_work['tlad'],
            self.planner_work['vgain'],
            self.n_next_points
        )

        # Skip some waypoints since they are very close together
        lookahead_point_global = deepcopy(lookahead_point_global[::self.skip_next_points, ...])

        poses_global = np.array(
            [
                obs['poses_x'][0],
                obs['poses_y'][0],
                0
            ],
            dtype=np.float32
        )
        lookahead_point = np.array(lookahead_point_global, dtype=np.float32)
        # lookahead_points_relative = (lookahead_point[:, :2] - poses_global).flatten()
        lookahead_points_relative = (lookahead_point - poses_global).flatten()
        action_planner = np.array([steering_angle, speed])
        logging.debug(f'action_planner {action_planner}')

        # Make sure that planner cannot generate actions that are not in the env's action space
        action_planner = np.clip(
            action_planner,
            a_min=self.observation_space['action_planner'].low,
            a_max=self.observation_space['action_planner'].high
        )

        obs['action_planner'] = action_planner
        obs['lookahead_points_relative'] = lookahead_points_relative
        self.planner_action = action_planner
        return obs

    def action(self, action):
        """Returns a modified action before :meth:`env.step` is called.

        Args:
            action (np.ndarray): the action that comes from the residual

        Returns:
            np.ndarray: the modified action that is combined with the planner's action
        """
        # Clip to ensure that combination is in env action space
        out = np.clip(
            action * self.action_scaling + self.planner_action,
            self.observation_space['action_planner'].low,
            self.observation_space['action_planner'].high
        )
        return out

    def step(self, *args, **kwargs):
        """Calls :meth:`env.step` and renders the environment."""
        out = super().step(*args, **kwargs)
        self.render()
        return out

    def reset(self, *args, **kwargs):
        """Calls :meth:`env.reset` and renders the environment."""
        out = super().reset(*args, **kwargs)
        self.render()
        return out

    def render(self):
        """Renders the environment."""
        if self.env.render_mode is not None:
            if self._render not in self.env.render_callbacks:
                self.env.add_render_callback(self._render)
            super().render()

    def _render(self, x):
        """Renders the environment."""
        return render_callback(x, self.planner)
