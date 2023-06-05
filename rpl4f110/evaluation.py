import pickle
import os
import csv
import torch
import numpy as np
from tqdm import tqdm


class Evaluator:
    """Evaluate the agent in the environment.

    Attributes:
        config (DictConfig): Config.
        envs (gym.Env): Environment.
        record_interval (int): Interval to record the evaluation.
        record_episodes (int): Number of episodes to record.
        maps (list): List of maps to evaluate on.
        _count (int): Number the evaluator has been called
    """

    def __init__(self, config, envs, maps, record_interval: int, record_episodes: int = 10) -> None:
        self.config = config
        self.envs = envs
        self.record_interval = record_interval
        self.record_episodes = record_episodes
        self.maps = maps
        self._count = 0

    @torch.inference_mode()
    def __call__(self, agent, obs_rms=None, global_step=None, force_eval=False, save_agent=False):
        """ Evaluate the agent in the environment.

        Args:
            agent (Agent): Agent to evaluate.
            obs_rms (RunningMeanStd): Running mean and std of the observations.
            global_step (int): Current global step.
            force_eval (bool): Force evaluation.
            save_agent (bool): Save the agent to disk.
        """

        self._count += 1
        if self._count % self.record_interval == 0 or force_eval:
            if obs_rms is not None:
                self.envs.set_obs_rms(obs_rms)

            if save_agent:
                save_dir = f'{self.config.log_dir}/eval/{global_step}/'
                os.makedirs(save_dir, exist_ok=True)
                torch.save(
                    {'model_state_dict': agent.state_dict(), 'obs_rms': self.envs.get_obs_rms()},
                    f'{save_dir}agent.pt'
                )
            print(f'\nEvaluation @ {global_step} global steps:')
            agent.train(False)
            self.envs.set_block_update_obs(True)  # Running stats should not be updated during eval runs
            self.vec_eval(self.envs, agent, global_step, map_name=self.maps)
            self.envs.set_block_update_obs(False)
            agent.train(True)

    def vec_eval(self, envs, agent, global_step=None, map_name=None):
        """ Evaluate the agent in the environment.

        Args:
            envs (gym.Env): Environment.
            agent (Agent): Agent to evaluate.
            global_step (int): Current global step.
            map_name (list): List of maps to evaluate on.

        Returns:
            records_all (list): List of records.
            avg_finish_time: Average finish time.
            n_crash: Number of crashes.
        """
        avg_finish_time = []
        n_crash = 0
        records_all = []

        histories = vec_rollout(envs, agent, self.config.sim.action_scaling, map_name, n_eval=self.record_episodes)
        print(
            f'\t\t\tfinish_time'
            f'\ta_r_vel_mean'
            f'\ta_r_vel_std'
            f'\ta_r_steer_mean'
            f'\ta_r_steer_std'
            f'\ttotal_r'
        )

        # Create records for all episodes and maps in the history
        for map_name_, history in zip(map_name, histories):
            history_ = history.history
            for episode, history_eps in enumerate(history_):
                obs_all = dict()
                for key in history_eps['obs'][0].keys():
                    # Flatten list into dicts
                    obs_all[key] = np.array([o[key] for o in history_eps['obs']])
                obs_all['reward'] = np.array(history_eps['reward'])[:-1, ...]  # Remove the np.nan in the last entry
                obs_all['action_residual'] = np.array(
                    history_eps['action'])[:-1, ...]  # Remove the np.nan in the last entry
                obs_all['action_planner'] = obs_all['action_planner'][:-1, ...]  # Remove the np.nan in the last entry
                obs_all['action_applied'] = np.array([o['prev_action'] for o in history_eps['obs']])[1:, ...]

                records = get_records_episode(obs_all)

                if not records["metrics"]['collisions']:
                    avg_finish_time.append(records["metrics"]["best_finish_time"])
                n_crash += records["metrics"]['collisions']
                records_all.append(records)

                print(
                    f'\t- {str(map_name_[:10])}:'
                    f'\t{np.round(records["metrics"]["best_finish_time"], 2):0.2f}'
                    f'\t\t{records["metrics"]["action_residual_vel_mean"]:0.2f}'
                    f'\t\t{records["metrics"]["action_residual_vel_std"]:0.2f}'
                    f'\t\t{records["metrics"]["action_residual_steer_mean"]:0.4f}'
                    f'\t\t{records["metrics"]["action_residual_steer_std"]:0.4f}'
                    f'\t\t{records["metrics"]["total_return"].round(2):.2f}'
                )

                log_dir = f'{self.config.log_dir}/eval/{global_step}/{map_name_}/{episode}/'
                os.makedirs(log_dir, exist_ok=True)
                with open(f'{log_dir}records.pkl', 'wb') as f:
                    pickle.dump(records, f, protocol=3)

                with open(f"{log_dir}history.csv", "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=records['history'].keys())
                    writer.writeheader()
                    for row in range(records['history']['action_residual_steer'].shape[0]):
                        # Note: Obs and action have different length, i.e., the real last obs (= when done) is missing
                        writer.writerow({k: v[row] for k, v in records['history'].items() if k != 'finish_times'})
                with open(f"{log_dir}metrics.csv", "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=records['metrics'].keys())
                    writer.writeheader()
                    writer.writerow(records['metrics'])

        return records_all, avg_finish_time, n_crash


class RolloutRecording:
    """List like object that stores a history of recorded episodes.

    Attributes:
        map_name (str): Name of the environment the episodes are recorded in
        history (list): Data storage of the recordings
    """

    def __init__(self, map_name):
        self.map_name = map_name
        self.history = [{'obs': [], 'action': [], 'reward': []}]

    def append(self, new_episode: bool, obs, action, reward):
        """Operator to append the recordings to the data storage."""
        self.history[-1]['obs'].append(obs)
        self.history[-1]['action'].append(action)
        self.history[-1]['reward'].append(reward)

        # The next set added will be the start of a new eps
        if new_episode:
            self.history.append({'obs': [], 'action': [], 'reward': []})


@torch.inference_mode()
def vec_rollout(envs, agent, action_scaler, map_name=None, n_eval=1):
    """Parallel rollout of the agent in vectorized envs.

    Note: Vector Env will reset itself automatically.

    Args:
        envs (VecEnv): Vectorized environment.
        agent (Agent): Agent to rollout.
        action_scaler (list): Scaling factor for the actions.
        map_name (list): List of map names.
        n_eval (int): Number of episodes to rollout.

    Returns:
        list: List of RolloutRecording objects, one for each map.
    """
    history = [RolloutRecording(n) for n in map_name]
    bar = tqdm(total=n_eval)

    obs_wrapped, infos = envs.reset()
    new_observations = infos['obs_']  # Access the non-normalized observations through infos
    while True:
        action = agent.get_mean_action(
            torch.tensor(obs_wrapped, dtype=torch.float32).to(agent.get_device())
        ).cpu().numpy()
        old_observations = new_observations
        obs_wrapped, rewards, terminateds, truncateds, infos = envs.step(action)
        new_observations = infos['obs_']

        for i in range(len(map_name)):
            history[i].append(
                False,
                old_observations[i],
                action[i, :] * action_scaler,
                rewards[i]
            )

        if 'final_info' in infos:
            for i in range(len(map_name)):
                if infos['_final_info'][i]:
                    history[i].append(
                        True,
                        infos['final_info'][i]['obs_'],
                        np.array([np.nan, np.nan]),
                        np.nan
                    )

        if all([len(h.history) > bar.n + 1 for h in history]):
            bar.update(1)
            if all([len(h.history) > n_eval for h in history]):
                for h in history:
                    h.history = h.history[:n_eval]
                break
    return history


def get_records_episode(obs_all):
    """Process the recorded observations to obtain statistics of them."""
    records = dict()

    # History of states
    history = dict()
    history['poses_x'] = obs_all['poses_x'].squeeze()
    history['poses_y'] = obs_all['poses_y'].squeeze()
    history['lookahead_points_relative_x'] = obs_all['lookahead_points_relative'][:, 0]
    history['lookahead_points_relative_y'] = obs_all['lookahead_points_relative'][:, 1]
    history['linear_vels_x'] = obs_all['linear_vels_x'].squeeze()
    history['linear_vels_y'] = obs_all['linear_vels_y'].squeeze()
    history['ang_vels'] = obs_all['ang_vels_z'].squeeze()
    history['slip_angle'] = obs_all['slip_angle'].squeeze()
    history['acc_x'] = obs_all['acc_x'].squeeze()
    history['acc_y'] = obs_all['acc_y'].squeeze()
    history['rewards'] = obs_all['reward'].squeeze()
    history['collisions'] = obs_all['collisions'].squeeze()
    history['lap_times'] = obs_all['lap_times'].squeeze()
    history['lap_counts'] = obs_all['lap_counts'].squeeze()
    history['finish_times'] = [history['lap_times'][history['lap_counts'] == i + 1].min() for i in
                               range(0, history['lap_counts'].max()) if
                               not any(history['collisions'][history['lap_counts'] == i + 1])]

    # # History of actions
    # Note: Len of action is 1 less since obs include the final 'done' state which doesn't require an action
    history['action_residual_steer'] = obs_all['action_residual'].squeeze()[:, 0]
    history['action_residual_vel'] = obs_all['action_residual'].squeeze()[:, 1]
    history['action_planner_steer'] = obs_all['action_planner'][:obs_all['action_residual'].shape[0], 0]
    history['action_planner_vel'] = obs_all['action_planner'][:obs_all['action_residual'].shape[0], 1]
    # Applied action is really the action thas been used in the env -> actions may be clipped!
    history['action_applied_steer'] = obs_all['action_applied'][:, 0]
    history['action_applied_vel'] = obs_all['action_applied'][:, 1]
    # Add to list
    records['history'] = history

    # History of performance
    metrics = dict()
    # Analysis of actions and stats
    for metrics_name in ['action_residual_vel', 'action_applied_vel', 'action_residual_steer', 'action_applied_steer',
                         'linear_vels_x']:
        metrics[f'{metrics_name}_mean'] = history[f'{metrics_name}'].mean(0)
        metrics[f'{metrics_name}_median'] = np.median(history[f'{metrics_name}'])
        metrics[f'{metrics_name}_std'] = history[f'{metrics_name}'].std(0)
        metrics[f'{metrics_name}_max'] = history[f'{metrics_name}'].max(0)
        metrics[f'{metrics_name}_min'] = history[f'{metrics_name}'].min(0)

    for metrics_name in ['action_residual_steer', 'action_applied_steer', 'linear_vels_y', 'slip_angle']:
        metrics[f'{metrics_name}_abs_mean'] = np.abs(history[f'{metrics_name}']).mean(0)
        metrics[f'{metrics_name}_abs_median'] = np.median(np.abs(history[f'{metrics_name}']))
        metrics[f'{metrics_name}_abs_std'] = np.abs(history[f'{metrics_name}']).std(0)
        metrics[f'{metrics_name}_abs_max'] = np.abs(history[f'{metrics_name}']).max(0)
        metrics[f'{metrics_name}_abs_min'] = np.abs(history[f'{metrics_name}']).min(0)

    # Performance metrics
    metrics['rewards_mean'] = history['rewards'].mean()
    metrics['rewards_std'] = history['rewards'].std()
    metrics['total_return'] = history['rewards'].sum()
    metrics['collisions'] = history['collisions'].sum(0)
    metrics['steps'] = history['rewards'].shape[0]  # 1 Step less than the num of observed states due to 'done'
    metrics['full_laps'] = history['lap_counts'].max()

    # Lap times are recorded for a maximum of 2 laps
    if len(history['finish_times']) < 2:
        metrics['best_finish_time'] = 0.0
    else:
        # Finish time is the lap times of the second lap (running start)
        metrics['best_finish_time'] = history['finish_times'][1] - history['finish_times'][0]

    # Add to list
    records['metrics'] = metrics

    return records
