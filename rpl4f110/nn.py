import torch
from torch import nn
from torch.distributions.normal import Normal
from torchrl.modules import TanhNormal
import numpy as np
from torchinfo import summary


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Layer initialization as in the PPO paper"""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def save_model_summary(config, models):
    """Save model summary to file.

    Args:
        config (DictConfig): Configuration dictionary.
        models (List[nn.Module]): List of models.
    """
    with open(f'{config.log_dir}/model_summary.txt', 'w') as file:
        for model in models:
            model_summary = summary(model, verbose=0)
            file.write(repr(model_summary) + '\n')


class LidarNetwork(nn.Sequential):
    """Network for processing lidar data

    Attributes:
        n_datapoints (torch.Tensor): Number of datapoints in the lidar data
        n_channels (torch.Tensor): Number of channels in the lidar data
    """

    def __init__(self, config):
        """Initializes the network with the given config.

        Note: Even when frames are stacked, only the last frame is used for the lidar data.
        """
        n_channels = 1  # config.env.frame_cat.n
        layers = (
            nn.AvgPool1d(3),  # [N, 8, 179]
            nn.Conv1d(n_channels, 16, 3, 2),  # [N, 8, 538]
            nn.ReLU(),
            nn.AvgPool1d(3),  # [N, 8, 179]
            nn.LazyConv1d(32, 3, 2),  # [N, 16, 89]
            nn.ReLU(),
            nn.AvgPool1d(3),  # [N, 16, 29]
            nn.Flatten()
        )
        super().__init__(*layers)
        self.register_buffer('n_datapoints', torch.tensor(1080))
        self.register_buffer('n_channels', torch.tensor(config.env.frame_cat.n))

    def reshape(self, data):
        """Reshapes the data to the correct shape for the network."""
        return data.view(-1, self.n_channels, self.n_datapoints)


class Agent(nn.Module):
    """Agent model implementing the PPO algorithm.

    Attributes:
        n_frame_stack (int): Number of frames stacked together.
        n_wpts (int): Number of waypoints used for the planner.
        lidar (nn.Module): Lidar network.
        critic (nn.Module): Critic network.
        actor_mean (nn.Module): Actor network for the mean.
        actor_logstd (nn.Parameter): Parameter for the standard deviation of the distribution.
        dist_head: Distribution head.
    """

    def __init__(self, config, envs):
        """Initializes the agent with the given config and environment."""
        super().__init__()

        self.n_frame_stack = int(config.env.frame_cat.n)
        self.n_wpts = int(3 * config.planner.n_next_points / config.planner.skip_next_points)

        self.lidar = LidarNetwork(config)
        self.critic = nn.Sequential(
            nn.LazyLinear(400),
            nn.ReLU(),
            layer_init(nn.Linear(400, 300)),
            nn.ReLU(),
            layer_init(nn.Linear(300, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            nn.LazyLinear(400),
            nn.ReLU(),
            layer_init(nn.Linear(400, 300)),
            nn.ReLU(),
            layer_init(nn.Linear(300, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))
        self.dist_head = TanhNormal if config.rl.distribution == 'TanhNormal' else Normal

    def get_device(self):
        """Return the device of the agent."""
        return next(self.actor_mean.parameters()).device

    def get_value(self, x):
        """Returns the value of the given state

        Args:
            x (torch.Tensor): Input tensor.
        """
        x_lidar, x_wpts, x_rest = self.split_data(x)

        hidden_lidar = self.lidar(x_lidar)
        x_critic = torch.cat([hidden_lidar, x_wpts, x_rest], -1)
        return self.critic(x_critic)

    def split_data(self, x):
        """Splits the input data into lidar, waypoints and rest of the states.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            x_lidar (torch.Tensor): Lidar data.
            x_wpts (torch.Tensor): Waypoints.
            x_rest (torch.Tensor): Rest of the states.
        """
        n_feats = int(x.shape[-1] / self.n_frame_stack)
        scan_points = 1080
        # Rearrange to so that the second dim is the FrameStack axis
        x_out = x.view(-1, self.n_frame_stack, n_feats)
        # x_lidar = x_out[:, -1, :1080].view(-1, 1, 1080)  # Lidar info
        # x_wpts = x_out[:, -1, 1080:1080 + self.n_wpts]  # Only get the most recent wpts
        # x_rest = x_out[:, :, 1080 + self.n_wpts:].flatten(1)  # Rest of the states
        # Note: Be very careful with the order of the features here, depends on the name/order of the observations
        # space in the environment
        x_lidar = x_out[:, -1, -n_feats:-n_feats + scan_points].view(-1, 1, scan_points)  # Most recent lidar info
        x_rest = x_out[:, :, -n_feats + scan_points: - self.n_wpts:].flatten(1)  # Rest of the states
        x_wpts = x_out[:, -1, -self.n_wpts:]  # Only get the most recent wpts
        return x_lidar, x_wpts, x_rest

    def get_action_and_value(self, x, action=None):
        """Returns the action and value of the given state.

        Args:
            x (torch.Tensor): Input tensor.
            action (torch.Tensor): Action tensor.

        Returns:
            action (torch.Tensor): Action tensor.
            log_prob (torch.Tensor): Log probability of the action.
            entropy (torch.Tensor): Entropy of the action; always set to 0 for simplicity.
            value (torch.Tensor): Value of the state.
        """
        x_lidar, x_wpts, x_rest = self.split_data(x)

        hidden_lidar = self.lidar(x_lidar)
        x_actor = torch.cat([hidden_lidar, x_wpts, x_rest], -1)
        x_critic = torch.cat([hidden_lidar, x_wpts, x_rest], -1)
        action_mean = self.actor_mean(x_actor)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = self.dist_head(action_mean, action_std)

        if action is None:
            action = probs.sample()

        log_prob = probs.log_prob(action)
        if log_prob.dim() > 1:
            log_prob = log_prob.sum(1)
        return action, log_prob, torch.zeros_like(log_prob), self.critic(x_critic)

    def get_mean_action(self, x):
        """Returns the mean action of the given state.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            action (torch.Tensor): Mean action tensor.
        """
        x_lidar, x_wpts, x_rest = self.split_data(x)
        hidden_lidar = self.lidar(x_lidar)
        x_actor = torch.cat([hidden_lidar, x_wpts, x_rest], -1)
        action_mean = self.actor_mean(x_actor)
        action = self.dist_head(action_mean, torch.zeros_like(action_mean)).sample()
        return action


class BaselineAgent:
    """Agent that outputs a zero residual action."""

    def get_device(self):
        """Return the device of the agent."""
        return 'cpu'

    def get_mean_action(self, x):
        """Returns a zero action."""
        # Rearrange to so that the second dim is the FrameStack axis
        x_out = x.view(-1, x.shape[-1])
        return torch.zeros(x_out.shape[0], 2)
