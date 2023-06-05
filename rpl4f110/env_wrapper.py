from copy import deepcopy
import gymnasium as gym
import numpy as np


class NormalizeObservation(gym.wrappers.NormalizeObservation):
    """Wrapper that can block the update of the running mean and variance of the observations.

    Attributes:
        _block_update_obs (bool): Variable to block the update of the running mean and variance of the observations.
    """

    def __init__(self, *args, **kwargs):
        """Initialises the wrapper."""
        super().__init__(*args, **kwargs)
        self._block_update_obs = False

    @property
    def block_update_obs(self):
        """Variable to block the update of the running mean and variance of the observations."""
        return self._block_update_obs

    def set_block_update_obs(self, value):
        """Blocks the update of the running mean and variance of the observations.

        Note: Property is not correctly used. This is a workaround so that this function is exposed in wrapped envs.
        """
        self._block_update_obs = value

    def get_obs_rms(self):
        """Returns the running mean and variance of the observations."""
        return deepcopy(self.obs_rms)

    def set_obs_rms(self, value):
        """Sets the running mean and variance of the observations."""
        self.obs_rms = value

    def normalize(self, obs):
        """Returns the normalised observation if not blocked."""
        if not self._block_update_obs:
            self.obs_rms.update(obs)
        return (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)


class NormalizeReward(gym.wrappers.NormalizeReward):
    """Wrapper that can block the update of the running mean and variance of the rewards.

    Attributes:
        _block_update_rew (bool): Variable to block the update of the running mean and variance of the rewards.
    """

    def __init__(self, *args, **kwargs):
        """Initialises the wrapper."""
        super().__init__(*args, **kwargs)
        self._block_update_rew = False

    @property
    def block_update_rew(self):
        """Variable to block the update of the running mean and variance of the rewards."""
        return self._block_update_rew

    def set_block_update_rew(self, value):
        """Blocks the update of the running mean and variance of the rewards.

        Note: Property is not correctly used. This is a workaround so that this function is exposed in wrapped envs.
        """
        self._block_update_rew = value

    def get_rew_rms(self):
        """Returns the running mean and variance of the rewards."""
        return deepcopy(self.return_rms)

    def set_rew_rms(self, value):
        """Sets the running mean and variance of the rewards."""
        self.return_rms = value

    def normalize(self, rews):
        """Returns the normalised reward if not blocked."""
        if not self._block_update_rew:
            self.return_rms.update(self.returns)
        return rews / np.sqrt(self.return_rms.var + self.epsilon)
