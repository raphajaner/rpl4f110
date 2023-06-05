import gymnasium as gym
from rpl4f110.planner import PlannerEnvWrapper


def make_env(config, map_names, index=0):
    """Create a gym environment with the given config and map names.

    Args:
        config: Config object
        map_names: List of map names to be used in the environment
        index: Index of the environment

    Returns:
        A function that creates a gym environment
    """

    map_names = [map_names] if type(map_names) is not list else map_names

    def thunk():
        env = gym.make(
            'f110_gym:f110-v1',
            config=config,
            map_names=map_names,
            render_mode=config.sim.render_mode if config.sim.render_mode != 'None' else None
        )

        env = PlannerEnvWrapper(env, config)  # Planner wrapper that includes the planner's action in the observation
        env = gym.wrappers.FilterObservation(env, filter_keys=config.env.obs_keys)
        env = gym.wrappers.FlattenObservation(env)
        if config.env.frame_cat.use:
            env = gym.wrappers.FrameStack(env, num_stack=config.env.frame_cat.n)
            env = gym.wrappers.FlattenObservation(env)  # Flatten stacked frames again

        return env

    return thunk
