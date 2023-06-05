import os
import random
import time
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
import numpy as np
import hydra
from hydra.core.hydra_config import HydraConfig, DictConfig
from omegaconf import OmegaConf
from tqdm import tqdm
import flatdict

from rpl4f110 import evaluation
from rpl4f110 import env_wrapper
from rpl4f110 import nn
from rpl4f110 import maker


@hydra.main(version_base=None, config_path='../configs/', config_name='config')
def main(config: DictConfig) -> None:
    """Main function to run the training.

    Args:
        config (DictConfig): Configuration object.
    """
    print(f'Starting a new run!\n'
          f'- Time: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}.\n'
          f'- Log_dir: {HydraConfig.get().run.dir}.')

    config['log_dir'] = HydraConfig.get().run.dir
    config['batch_size'] = int(config.rl.num_envs * config.rl.num_steps)
    config['minibatch_size'] = int(config.batch_size // config.rl.num_minibatches)
    config['num_updates'] = int(config.rl.total_timesteps // config.batch_size)
    OmegaConf.save(config, os.path.join(config.log_dir, "config.yaml"))

    # Tensorboard for logging
    writer = SummaryWriter(config.log_dir)
    dict_config = flatdict.FlatDict(OmegaConf.to_container(config))
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in dict_config.items()])),
    )

    # Seeding
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = config.torch_deterministic

    if config.cuda_allow_tf32:
        # Get some speed-up by using TensorCores on Nvidia Ampere GPUs
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

    device = torch.device(f"cuda:{config.gpu_id}" if torch.cuda.is_available() and config.cuda else "cpu")

    # Select if parallel env runs async or not
    gym_vector_cls = gym.vector.AsyncVectorEnv if config.async_env else gym.vector.SyncVectorEnv

    # Training envs + wrapper (obs and rew)
    assert config.rl.num_envs % len(config.maps.maps_train) == 0, "Number of envs must be divisible by number of maps."
    maps_train = list(config.maps.maps_train) * int(config.rl.num_envs / len(config.maps.maps_train))
    envs = gym_vector_cls([maker.make_env(config, maps_train[i], i) for i in range(len(maps_train))])
    if config.env.wrapper.normalize_obs:
        envs = env_wrapper.NormalizeObservation(envs)
    if config.env.wrapper.clip_obs != "None":
        envs = gym.wrappers.TransformObservation(
            envs, lambda obs: np.clip(obs, config.env.wrapper.clip_obs[0], config.env.wrapper.clip_obs[1])
        )
    if config.env.wrapper.normalize_rew:
        envs = env_wrapper.NormalizeReward(envs, gamma=config.rl.gamma)
    if config.env.wrapper.clip_rew != "None":
        envs = gym.wrappers.TransformReward(
            envs, lambda reward: np.clip(reward, config.env.wrapper.clip_rew[0], config.env.wrapper.clip_rew[1])
        )

    # Evaluation envs + wrapper(obs but no rew)
    maps_all = (config.maps.maps_train + config.maps.maps_test)
    envs_eval = gym.vector.AsyncVectorEnv([maker.make_env(config, maps_all[i], 0) for i in range(len(maps_all))])
    if config.env.wrapper.normalize_obs:
        envs_eval = env_wrapper.NormalizeObservation(envs_eval)
    if config.env.wrapper.clip_obs != "None":
        envs_eval = gym.wrappers.TransformObservation(
            envs_eval, lambda obs: np.clip(obs, config.env.wrapper.clip_obs[0], config.env.wrapper.clip_obs[1])
        )

    evaluator = evaluation.Evaluator(
        config,
        envs_eval,
        maps=(config.maps.maps_train + config.maps.maps_test),
        record_episodes=config.eval_n_eps,
        record_interval=config.eval_freq
    )

    # Allows to get some reference values of the baseline without residual actions applied
    if config.bench_baseline:
        from rpl4f110.nn import BaselineAgent
        # Baseline planner passes planned action through to env
        baseline = BaselineAgent()
        evaluator(baseline, force_eval=True)
        exit(0)

    # Init of the agent (actor and value network)
    agent = nn.Agent(config, envs).to(device)

    # Init lazy layers
    with torch.no_grad():
        next_obs, _ = envs.reset(seed=config.seed)
        agent.get_action_and_value(torch.Tensor(next_obs).to(device))
    nn.save_model_summary(config, models=[agent])

    if config.inference:
        data = torch.load(f'{config.inference_log}agent.pt', map_location=device)
        agent.load_state_dict(data['model_state_dict'])
        agent = agent.to(device)
        evaluator(agent, obs_rms=data['obs_rms'], force_eval=True)
        exit(0)

    optimizer = torch.optim.Adam(
        agent.parameters(),
        lr=config.rl.learning_rate, eps=1e-5
    )

    if config.rl.anneal_lr:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.num_updates + 1,
            eta_min=config.rl.anneal_lr_factor * optimizer.param_groups[0]["lr"]
        )

    # ALGO Logic: Storage setup
    obs = torch.zeros((config.rl.num_steps, config.rl.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((config.rl.num_steps, config.rl.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((config.rl.num_steps, config.rl.num_envs)).to(device)
    rewards = torch.zeros((config.rl.num_steps, config.rl.num_envs)).to(device)
    dones = torch.zeros((config.rl.num_steps, config.rl.num_envs)).to(device)
    terminateds = torch.zeros((config.rl.num_steps, config.rl.num_envs)).to(device)
    truncateds = torch.zeros((config.rl.num_steps, config.rl.num_envs)).to(device)
    values = torch.zeros((config.rl.num_steps, config.rl.num_envs)).to(device)

    # Start learning
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=config.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(config.rl.num_envs).to(device)
    next_terminated = torch.zeros(config.rl.num_envs).to(device)
    next_truncated = torch.zeros(config.rl.num_envs).to(device)

    for update in tqdm(range(1, config.num_updates + 1)):
        if config.env.wrapper.block_updates and update > config.env.wrapper.block_after_n_updates:
            envs.set_block_update_rew(True)
            envs.set_block_update_obs(True)
            assert envs.block_update_rew and envs.block_update_obs, "Updating normalization stats couldn't be blocked."

        for step in range(0, config.rl.num_steps):
            global_step += 1 * config.rl.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            terminateds[step] = next_terminated
            truncateds[step] = next_truncated
            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
            next_terminated = torch.Tensor(terminated).to(device)
            next_truncated = torch.Tensor(truncated).to(device)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(config.rl.num_steps)):
                if t == config.rl.num_steps - 1:
                    nextnonterminal = 1.0 - next_terminated
                    nextvalues = next_value
                    nextnonedone = 1.0 - next_done
                else:
                    nextnonterminal = 1.0 - terminateds[t + 1]
                    nextvalues = values[t + 1]
                    nextnonedone = 1.0 - dones[t + 1]
                delta = rewards[t] + config.rl.gamma * nextvalues * nextnonterminal - values[t]
                advantages[
                    t] = lastgaelam = delta + config.rl.gamma * config.rl.gae_lambda * nextnonedone * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(config.batch_size)
        clipfracs = []
        for epoch in range(config.rl.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, config.batch_size, config.minibatch_size):
                end = start + config.minibatch_size
                mb_inds = b_inds[start:end]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > config.rl.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if config.rl.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - config.rl.clip_coef, 1 + config.rl.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if config.rl.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -config.rl.clip_coef,
                        config.rl.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - config.rl.ent_coef * entropy_loss + v_loss * config.rl.vf_coef

                optimizer.zero_grad()
                loss.backward()
                grads = torch.nn.utils.clip_grad_norm_(agent.parameters(), config.rl.max_grad_norm)
                optimizer.step()

            if config.rl.target_kl != 'None':
                if approx_kl > config.rl.target_kl:
                    break
        if config.rl.anneal_lr:
            scheduler.step()

        # Batch stats
        writer.add_scalar("batch/rewards_mean", rewards.mean().item(), global_step)
        writer.add_scalar("batch/rewards_std", rewards.std().item(), global_step)
        writer.add_scalar("batch/rewards_min", rewards.min().item(), global_step)
        # Loss stats
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("losses/epochs", epoch, global_step)
        writer.add_scalar("losses/grads", grads, global_step)
        writer.add_scalar("losses/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        # Others
        writer.add_scalar("others/SPS", int(global_step / (time.time() - start_time)), global_step)
        writer.add_scalar("others/actor_logstd[0]", torch.exp(agent.actor_logstd)[0, 0].item(), global_step)
        writer.add_scalar("others/actor_logstd[1]", torch.exp(agent.actor_logstd)[0, 1].item(), global_step)

        # Evaluate performance every N steps
        evaluator(agent, global_step=global_step, obs_rms=envs.get_obs_rms(), save_agent=True)

    # Final evaluation
    evaluator(agent, global_step=global_step, force_eval=True, obs_rms=envs.get_obs_rms(), save_agent=True)

    # Clean up
    envs.close()
    writer.close()
