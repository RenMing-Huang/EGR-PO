import os
import time
from datetime import datetime
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from absl import app, flags
from functools import partial
import numpy as np
from src.agents import hdql as learner

from src import d4rl_utils, d4rl_ant, ant_diagnostics, viz_utils
from src.gc_dataset import GCSDataset

from rl_m.wandb import setup_wandb, default_wandb_config
import wandb
from rl_m.evaluation import supply_rng, evaluate_with_trajectories, EpisodeMonitor
import gzip
from ml_collections import config_flags
import pickle

from src.utils import record_video
from src.agents.hdql import device

from sac_agent import Guided_SAC_countinuous 

FLAGS = flags.FLAGS
flags.DEFINE_string('env_name', 'antmaze-large-play-v2', '')
flags.DEFINE_string('save_dir', f'experiment_output/', '')
flags.DEFINE_string('run_group', 'Debug', '')
flags.DEFINE_integer('seed', 0, '')
flags.DEFINE_integer('eval_episodes', 50, '')
flags.DEFINE_integer('num_video_episodes', 2, '')
flags.DEFINE_integer('log_interval', 1000, '')
flags.DEFINE_integer('eval_interval', 100000, '')
flags.DEFINE_integer('save_interval', 100000, '')
flags.DEFINE_integer('batch_size', 1024, '')
flags.DEFINE_integer('pretrain_steps', 0, '')

flags.DEFINE_integer('use_layer_norm', 1, '')
flags.DEFINE_integer('value_hidden_dim', 512, '')
flags.DEFINE_integer('value_num_layers', 3, '')
flags.DEFINE_integer('use_rep', 0, '')
flags.DEFINE_integer('rep_dim', None, '')
flags.DEFINE_integer('goal_dim', None, '')
flags.DEFINE_enum('rep_type', 'state', ['state', 'diff', 'concat'], '')
flags.DEFINE_integer('policy_train_rep', 0, '')
flags.DEFINE_integer('use_waypoints', 0, '')
flags.DEFINE_integer('way_steps', 1, '')

flags.DEFINE_float('pretrain_expectile', 0.7, '')
flags.DEFINE_float('p_randomgoal', 0.3, '')
flags.DEFINE_float('p_trajgoal', 0.5, '')
flags.DEFINE_float('p_currgoal', 0.2, '')
flags.DEFINE_float('high_p_randomgoal', 0., '')
flags.DEFINE_integer('geom_sample', 1, '')
flags.DEFINE_float('discount', 0.99, '')
flags.DEFINE_float('temperature', 1, '')
flags.DEFINE_float('high_temperature', 1, '')

flags.DEFINE_integer('visual', 0, '')
flags.DEFINE_string('encoder', 'impala', '')

flags.DEFINE_string('algo_name', None, '')  # Not used, only for logging

flags.DEFINE_integer('device', 0, '')
flags.DEFINE_float('a_eta', 0.5, '')
flags.DEFINE_float('ha_eta', 0.5, '')
flags.DEFINE_string('pretrain_path', None, '')
flags.DEFINE_string('ele_path', None, '')
flags.DEFINE_string('load_path', None, '')
flags.DEFINE_string('pred_goal', None, '')


wandb_config = default_wandb_config()
wandb_config.update({
    'project': 'diffusion_guided_sac',
    'group': 'Debug',
    'name': '{env_name}',
})

config_flags.DEFINE_config_dict('wandb', wandb_config, lock_config=False)
config_flags.DEFINE_config_dict('config', learner.get_default_config(), lock_config=False)

gcdataset_config = GCSDataset.get_default_config()
config_flags.DEFINE_config_dict('gcdataset', gcdataset_config, lock_config=False)

def kitchen_render(kitchen_env, wh=64):
    from dm_control.mujoco import engine
    camera = engine.MovableCamera(kitchen_env.sim, wh, wh)
    camera.set_pose(distance=1.8, lookat=[-0.3, .5, 2.], azimuth=90, elevation=-60)
    img = camera.render()
    return img

def get_debug_statistics(agent, batch):
    def get_info(s, g):
        return agent.network.value(s, g, info=True)

    s = batch['observations']
    g = batch['goals']

    info = get_info(s, g)

    stats = {}

    stats.update({
        'v': info['v'].mean(),
    })

    return stats


def get_gcvalue(agent, s, g):
    s = torch.from_numpy(s).float().to(device)
    g = torch.from_numpy(g).float().to(device)
    v1, v2 = agent.network.value(s, g)
    return (v1 + v2) / 2


def get_v(agent, goal, observations):
    goal = np.tile(goal, (observations.shape[0], 1))
    return get_gcvalue(agent, observations, goal)



def get_traj_v(agent, trajectory):
    def get_v(s, g):
        s = torch.from_numpy(s).float().to(device)
        g = torch.from_numpy(g).float().to(device)
        v1, v2 = agent.network.value(s[None], g[None])
        return (v1 + v2) / 2
    
    observations = trajectory['observations']
    all_values = np.zeros((observations.shape[0], observations.shape[0]))
    
    for i in range(observations.shape[0]):
        for j in range(observations.shape[0]):
            all_values[i, j] = get_v(observations[i], observations[j])
    
    return {
        'dist_to_beginning': all_values[:, 0,],
        'dist_to_end': all_values[:, -1],
        'dist_to_middle': all_values[:, all_values.shape[1] // 2]
    }

def get_distance(v):
    gamma=0.99
    log_gamma = np.log(gamma)
    one_minus_gamma = 1 - gamma
    value = np.clip(one_minus_gamma * v, -0.95, None)
    return np.log(1.0 + value) / log_gamma

@torch.no_grad()
def get_guide_v(next_observation, goal, value_fn):
    next_observation = torch.from_numpy(next_observation).float().unsqueeze(0).to(device)
    if isinstance(goal, np.ndarray):
        goal = torch.from_numpy(goal).float().unsqueeze(0).to(device)

    v1, v2 = value_fn(next_observation, goal)
    v = (v1 + v2) / 2
    return v.item()

@torch.no_grad()
def get_reward(state, next_state, goal, value_fn):
    state = torch.from_numpy(state).float().reshape(1, -1).to(device)
    next_state = torch.from_numpy(next_state).float().reshape(1, -1).to(device)
    goal = torch.from_numpy(goal).float().reshape(1, -1).to(device)
    v1, v2 = value_fn(state, goal)
    _v1, _v2 = value_fn(next_state, goal)
    v = (v1 + v2) / 2
    _v = (_v1 + _v2) / 2

    # scale to [-1, 1]
    advantage = (_v - v).item()
    # if advantage > 1 or advantage < -1:
    #     advantage = 0
    advantage = 2 / (1 + np.exp(-advantage)) - 1

    return advantage

def reward_adapt(reward, env_name):
    if isinstance(reward, np.ndarray):
        reward = reward[0]
    if 'Sawyer' in env_name:
        reward += 1
    elif 'Fetch' in env_name:
        reward = reward + 1

    return reward

def rewards_fn(env_name):
    def distance_based_reward(s, goal, threshold=0.5, dim_range=[0, 2]):
        s = s[:, dim_range[0]:dim_range[1]]
        goal = goal[:, dim_range[0]:dim_range[1]]
        return np.linalg.norm(s - goal, axis=-1) < threshold
    
    if 'antmaze' in env_name:
        return partial(distance_based_reward, threshold=0.5, dim_range=[0, 2])
    elif 'FetchReach' in env_name:
        return partial(distance_based_reward, threshold=0.05, dim_range=[0, 3])
    return None

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

def main(_):
    # set_seed(FLAGS.seed)
    g_start_time = int(datetime.now().timestamp())
    str_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    exp_name = ''
    # exp_name += f'sd{FLAGS.seed:03d}_'
    if 'SLURM_JOB_ID' in os.environ:
        exp_name += f's_{os.environ["SLURM_JOB_ID"]}.'
    if 'SLURM_PROCID' in os.environ:
        exp_name += f'{os.environ["SLURM_PROCID"]}.'
    if 'SLURM_RESTART_COUNT' in os.environ:
        exp_name += f'rs_{os.environ["SLURM_RESTART_COUNT"]}.'
    exp_name += f'{str_time}'
    exp_name += f'_{FLAGS.wandb["name"]}'

    FLAGS.gcdataset['p_randomgoal'] = FLAGS.p_randomgoal
    FLAGS.gcdataset['p_trajgoal'] = FLAGS.p_trajgoal
    FLAGS.gcdataset['p_currgoal'] = FLAGS.p_currgoal
    FLAGS.gcdataset['geom_sample'] = FLAGS.geom_sample
    FLAGS.gcdataset['high_p_randomgoal'] = FLAGS.high_p_randomgoal
    FLAGS.gcdataset['way_steps'] = FLAGS.way_steps
    FLAGS.gcdataset['discount'] = FLAGS.discount
    FLAGS.config['pretrain_expectile'] = FLAGS.pretrain_expectile
    FLAGS.config['discount'] = FLAGS.discount
    FLAGS.config['temperature'] = FLAGS.temperature
    FLAGS.config['high_temperature'] = FLAGS.high_temperature
    FLAGS.config['use_waypoints'] = FLAGS.use_waypoints
    FLAGS.config['way_steps'] = FLAGS.way_steps
    FLAGS.config['value_hidden_dims'] = (FLAGS.value_hidden_dim,) * FLAGS.value_num_layers
    FLAGS.config['use_rep'] = FLAGS.use_rep
    FLAGS.config['rep_dim'] = FLAGS.rep_dim
    FLAGS.config['policy_train_rep'] = FLAGS.policy_train_rep
    FLAGS.config['a_eta'] = FLAGS.a_eta
    FLAGS.config['ha_eta'] = FLAGS.ha_eta

    # Create wandb logger
    params_dict = {**FLAGS.gcdataset.to_dict(), **FLAGS.config.to_dict()}
    FLAGS.wandb['name'] = FLAGS.wandb['exp_descriptor'] = exp_name
    FLAGS.wandb['group'] = FLAGS.wandb['exp_prefix'] = FLAGS.run_group
    setup_wandb(params_dict, **FLAGS.wandb)

    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, wandb.config.exp_prefix, wandb.config.experiment_id)
    os.makedirs(FLAGS.save_dir, exist_ok=True)

    discrete = False

    env_name = FLAGS.env_name
    if 'antmaze' in env_name:
        if 'ultra' in FLAGS.env_name:
            import d4rl_ext
            import gym
            env = gym.make(env_name)
            env = EpisodeMonitor(env)
        else:
            env = d4rl_utils.make_env(env_name)
        
        eval_env = d4rl_utils.make_env(env_name)
        eval_env.render(mode='rgb_array', width=200, height=200)
        if 'large' in FLAGS.env_name:
            eval_env.viewer.cam.lookat[0] = 18
            eval_env.viewer.cam.lookat[1] = 12
            eval_env.viewer.cam.distance = 50
            eval_env.viewer.cam.elevation = -90

        elif 'ultra' in FLAGS.env_name:
            eval_env.viewer.cam.lookat[0] = 26
            eval_env.viewer.cam.lookat[1] = 18
            eval_env.viewer.cam.distance = 70
            eval_env.viewer.cam.elevation = -90
        else:
            eval_env.viewer.cam.lookat[0] = 18
            eval_env.viewer.cam.lookat[1] = 12
            eval_env.viewer.cam.distance = 50
            eval_env.viewer.cam.elevation = -90
        dataset = d4rl_utils.get_dataset(env, FLAGS.env_name)
        dataset = dataset.copy({'rewards': dataset['rewards'] - 1.0})
    elif 'kitchen' in env_name:
        env = d4rl_utils.make_env(FLAGS.env_name)
        eval_env = d4rl_utils.make_env(FLAGS.env_name)
        dataset = d4rl_utils.get_dataset(env, FLAGS.env_name, filter_terminals=True)
        dataset = dataset.copy({'observations': dataset['observations'][:, :30], 'next_observations': dataset['next_observations'][:, :30]})
    elif 'FetchReach' in env_name:
        env = d4rl_utils.make_env(env_name)
        obs_dim = env.observation_space['observation'].shape[0]
        eval_env = d4rl_utils.make_env(env_name)
        dataset = d4rl_utils.create_dataset(FLAGS.load_path, obs_dim)
    elif 'maze2d' in env_name:
        env = d4rl_utils.make_env(env_name)
        eval_env = d4rl_utils.make_env(env_name)
        dataset = d4rl_utils.get_dataset(env, FLAGS.env_name)
    elif 'calvin' in FLAGS.env_name:
        from src.envs.calvin import CalvinEnv
        from hydra import compose, initialize
        from src.envs.gym_env import GymWrapper
        from src.envs.gym_env import wrap_env
        initialize(config_path='src/envs/conf')
        cfg = compose(config_name='calvin')
        env = CalvinEnv(**cfg)
        env.max_episode_steps = cfg.max_episode_steps = 360
        env = GymWrapper(
            env=env,
            from_pixels=cfg.pixel_ob,
            from_state=cfg.state_ob,
            height=cfg.screen_size[0],
            width=cfg.screen_size[1],
            channels_first=False,
            frame_skip=cfg.action_repeat,
            return_state=False,
        )
        env = wrap_env(env, cfg)
        from copy import deepcopy
        eval_env = deepcopy(env)
        data = pickle.load(gzip.open('data/calvin.gz', "rb"))
        ds = []
        for i, d in enumerate(data):
            if len(d['obs']) < len(d['dones']):
                continue  # Skip incomplete trajectories.
            # Only use the first 21 states of non-floating objects.
            d['obs'] = d['obs'][:, :21]
            new_d = dict(
                observations=d['obs'][:-1],
                next_observations=d['obs'][1:],
                actions=d['actions'][:-1],
            )
            num_steps = new_d['observations'].shape[0]
            new_d['rewards'] = np.zeros(num_steps)
            new_d['terminals'] = np.zeros(num_steps, dtype=bool)
            new_d['terminals'][-1] = True
            ds.append(new_d)
        dataset = dict()
        for key in ds[0].keys():
            dataset[key] = np.concatenate([d[key] for d in ds], axis=0)
        dataset = d4rl_utils.get_dataset(None, FLAGS.env_name, dataset=dataset)
    # elif 'Hand' in FLAGS.env_name:
    #     import gymnasium as gym
    #     env = gym.make(FLAGS.env_name, render_mode='rgb_array')
    #     eval_env = gym.make(FLAGS.env_name, render_mode='rgb_array')
    #     obs_dim = env.observation_space['observation'].shape[0]
    #     assert FLAGS.load_path is not None
    #     dataset = d4rl_utils.create_dataset(FLAGS.load_path, obs_dim)
    elif 'Sawyer' in FLAGS.env_name:
        from wgcsl.common.env_util import get_env_type, build_env, get_game_envs
        _game_envs = get_game_envs()
        env_type, env_id = get_env_type(FLAGS.env_name, _game_envs)
        print('env_type: {}'.format(env_type))
        env = build_env(FLAGS.env_name, _game_envs)
        eval_env = build_env(FLAGS.env_name, _game_envs)
        obs_dim = env.observation_space['observation'].shape[0]
        dataset = d4rl_utils.create_dataset(FLAGS.load_path, obs_dim)
    else:
        env = d4rl_utils.make_env(env_name)
        obs_dim = env.observation_space['observation'].shape[0]
        eval_env = d4rl_utils.make_env(env_name)        
        dataset = d4rl_utils.create_dataset(FLAGS.load_path, obs_dim)

    pretrain_dataset = GCSDataset(FLAGS.env_name,dataset, **FLAGS.gcdataset.to_dict())
    example_batch = dataset.sample(1)
    agent = learner.create_learner(FLAGS.seed,
                                   example_batch['observations'],
                                   example_batch['actions'],
                                   visual=FLAGS.visual,
                                   encoder=FLAGS.encoder,
                                   discrete=discrete,
                                   use_layer_norm=FLAGS.use_layer_norm,
                                   rep_type=FLAGS.rep_type,
                                   goal_dim=FLAGS.goal_dim,
                                   **FLAGS.config)
    with open(FLAGS.pretrain_path, 'rb') as f:
        params = pickle.load(f)

    agent.network.load_state_dict(params['agent'])
    agent.network.to(device)
    agent.eval()

    if ('antmaze' in env_name) or ('maze2d' in env_name):
        observation_dim = env.observation_space.shape[0]
    elif 'kitchen' in env_name:
        observation_dim = dataset['observations'].shape[-1]
    elif ('Fetch' in env_name) or ('Sawyer' in env_name):
        observation_dim = env.observation_space['observation'].shape[0]


    if 'calvin' in env_name:
        example_batch = dataset.sample(1)
        observation_dim = example_batch['observations'].shape[-1]
        action_dim = example_batch['actions'].shape[-1]
    else:
        action_dim = env.action_space.shape[0]


    high_policy_fn = partial(agent.sample_high_actions, return_mean=True, num_samples=32)
    base_observation = np.array(pretrain_dataset.dataset['observations'][0])
    value_fn = partial(agent.network.value, encoded=True)
    goal_encoder = agent.network.value_goal_encoder
    try:
        max_steps = env.env._max_episode_steps
    except:
        max_steps = 360
    if 'Sawyer' in env_name:
        max_steps = 50

    sac_agent = Guided_SAC_countinuous(state_dim=observation_dim,
                                       action_dim=action_dim,
                                       dvc=device,
                                       adaptive_alpha=False,
                                       alpha=0.04,
                                       goal_encoder=goal_encoder,
                                       max_timesteps=max_steps,
                                       rep_dim=  FLAGS.rep_dim if FLAGS.use_rep else observation_dim,
                                       way_steps=FLAGS.way_steps,
                                       rewards_fn = rewards_fn(env_name),
                                       ag_dim = observation_dim,
                                       )
    
    # sac_agent.load(env_name, 500000)
    # for i in range(100):
    #     eval_ep_r, eval_ep_s, renders = evaluate_policy(env_name, eval_env, sac_agent, high_policy_fn, goal_encoder, value_fn, turns=30)
    #     print(f'Episode Reward:{eval_ep_r:.4f}, Episode Steps:{eval_ep_s:.1f}')
    #     video = record_video('Video', i, renders=renders[:2], skip_frames=3)
    #     wandb.log({'Video': video}, step=i)
    #     wandb.log({'reward': eval_ep_r}, step=i)
    #     wandb.log({'steps': eval_ep_s}, step=i)
    # exit()

    max_e_steps = max_steps
    update_every = 20
    eval_interval = 20 * max_e_steps
    save_interval = 200 * max_e_steps
    
    train_success, train_rewards = [], []
    total_steps = 0
    # if 'antmaze' in FLAGS.env_name:
    #     sac_agent.replay_buffer.sav_rep(FLAGS.env_name)
    online_finetune = False
    if online_finetune:
        # create empty online dataset
        from copy import deepcopy
        new_dataset = deepcopy(dataset)
        online_dataset = GCSDataset(FLAGS.env_name, new_dataset, **FLAGS.gcdataset.to_dict())
        online_dataset.empty_data()

    for i in range(10_000):
        # if total_steps > 100_000:
        #     if 'antmaze' in FLAGS.env_name:
        #         sac_agent.replay_buffer.sav_rep(FLAGS.env_name)
        if total_steps > 1_000_000:
            break
        rewards = 0
        extra_rewards = 0

        observation, done = env.reset(), False
        if 'antmaze' in env_name:
            goal = env.wrapped_env.target_goal
            obs_goal = base_observation.copy()
            obs_goal[:2] = goal
        elif 'kitchen' in env_name:
            observation, obs_goal = observation[:30], observation[30:]
            obs_goal[:9] = base_observation[:9]
        elif 'maze2d' in env_name:
            goal = env.get_target()
            obs_goal = np.array(goal)
            # obs_goal = np.zeros_like(base_observation)
            # obs_goal[:2] = goal
        elif ('FetchReach' in env_name):
            goal = observation['desired_goal']
            observation = observation['observation']
            obs_goal = np.zeros_like(base_observation)
            obs_goal[0:3] = goal
        elif  ('FetchPick' in env_name): 
            goal = observation['desired_goal']
            observation = observation['observation']
            obs_goal = np.zeros_like(base_observation)
            obs_goal[0:6] = np.concatenate([goal, goal])
        elif 'calvin' in env_name:
            observation = observation['ob']
            goal = np.array([0.25, 0.15, 0, 0.088, 1, 1])
            obs_goal = base_observation.copy()
            obs_goal[15:21] = goal
        elif 'Sawyer' in env_name:
            goal = observation['desired_goal'][0]
            observation = observation['observation']
            obs_goal = np.zeros_like(base_observation)
            obs_goal[-goal.shape[0]:] = goal

        loss = None
        ep_steps = 0
        reward = 0

        ep_obs, ep_sg, ep_actions, ep_ir, ep_r, ep_ag, ep_done = [], [], [], [], [], [], []
        
        obs_goal = torch.from_numpy(obs_goal).float().reshape(1, -1).to(device)
        while not done:
            total_steps += 1
            observation = observation.squeeze()
            with torch.no_grad():
                cur_obs_goal = high_policy_fn(observations=observation, goals=obs_goal, temperature=1)
                cur_obs_goal = cur_obs_goal.squeeze(0).cpu().numpy()
            if FLAGS.use_rep:
                cur_obs_goal = cur_obs_goal / np.linalg.norm(cur_obs_goal, axis=-1, keepdims=True) * np.sqrt(cur_obs_goal.shape[-1]).item()
            elif FLAGS.pred_goal:
                cur_obs_goal = cur_obs_goal + observation

            if (reward == 1.0) and 'maze2d' in env_name:
                cur_obs_goal[:2] = goal
                cur_obs_goal[2:] = 0
                
            cur_obs_goal_rep = cur_obs_goal
            

            action = sac_agent.select_action(observation, cur_obs_goal_rep, deterministic=False)

            # if (reward == 1.0) and ('Fetch' in env_name):
            #     action[:3] = 0

            if 'calvin' in env_name:
                next_observation, reward, done, info = env.step({'ac': np.array(action)})
                next_observation = next_observation['ob']
                del info['robot_info']
                del info['scene_info']
            else:
                next_observation, reward, done, info = env.step(action)

            if isinstance(next_observation, dict):
                next_observation = next_observation['observation']

            if 'kitchen' in env_name:
                next_observation = next_observation[:30]

            reward = reward_adapt(reward, env_name)

            # get intrinsic reward
            if 'maze2d' in env_name:
                i_r = get_reward(observation, next_observation, cur_obs_goal_rep[:2], value_fn)
            else:
                i_r = get_reward(observation, next_observation, cur_obs_goal_rep, value_fn)

            extra_rewards += i_r
            rewards += reward

            

            ep_obs.append(observation)
            ep_sg.append(cur_obs_goal_rep)
            ep_actions.append(action)
            ep_ir.append(i_r)
            ep_r.append(reward)
            ep_done.append(done)
            ep_ag.append(observation)



            ep_steps += 1

            if (total_steps > 5 * max_e_steps)  and (total_steps % update_every == 0):
                for _ in range(update_every):
                    loss = sac_agent.HER_train()
                # online finetune high-level policy
            if (total_steps > 5 * max_e_steps)  and (total_steps % (update_every) == 0):
                if online_finetune:
                    agent.train()
                    pretrain_batch = pretrain_dataset.sample(FLAGS.batch_size * 3 // 4)
                    online_batch = online_dataset.sample(FLAGS.batch_size // 4 )

                    # merge batch
                    for key in pretrain_batch.keys():
                        pretrain_batch[key] = np.concatenate([pretrain_batch[key], online_batch[key]], axis=0)
                    agent, online_finetune_loss, info = agent.pretrain_update(pretrain_batch)
                    agent.eval()
                    wandb.log(online_finetune_loss, step=total_steps)
                    wandb.log(info, step=total_steps)



            if (total_steps >= 20 * max_e_steps) and  (total_steps % eval_interval == 0):
                eval_ep_r, eval_ep_s, renders = evaluate_policy(env_name, eval_env, sac_agent, high_policy_fn, goal_encoder, value_fn, turns=50, base_observation=base_observation, pred_goal=FLAGS.pred_goal)

                print(f'Steps: {int(total_steps/1000)}k, Episode Reward:{eval_ep_r:.4f}, Episode Steps:{eval_ep_s:.1f}')
                if not 'Sawyer' in env_name:
                    video = record_video('Video', total_steps, renders=renders[:2], skip_frames=1)
                    wandb.log({'eval/Video': video}, step=total_steps)

                wandb.log({'eval/rewards': eval_ep_r}, step=total_steps)
                wandb.log({'eval/steps': eval_ep_s}, step=total_steps)
                wandb.log({'train/success': np.mean(train_success)}, step=total_steps)
                wandb.log({'train/rewards': np.mean(train_rewards)}, step=total_steps)
                train_success, train_rewards = [], []
            
            # if total_steps == 1 or total_steps % save_interval == 0:
            #     sac_agent.save(env_name, total_steps)
            #     print(f'Save model at {total_steps} steps')

            observation = next_observation.squeeze()

        train_success.append(1 if reward == 1 else 0)
        train_rewards.append(rewards)

        ep_obs.append(observation)
        if online_finetune:
            online_dataset.add_data({
                    'observations': np.array(ep_obs[:-1]),
                    'actions': np.array(ep_actions),
                    'rewards': np.array(ep_r),
                    'masks': 1 - np.array(ep_done),
                    'next_observations': np.array(ep_obs[1:]),
                    'dones_float': np.array(ep_done),
                    'achieved_goals': np.array(ep_obs[:-1]),
                    'next_achieved_goals': np.array(ep_obs[1:]),
                })
        sac_agent.replay_buffer.add_episode(ep_obs, ep_sg, ep_ir, ep_actions, ep_r, ep_ag, ep_done)

        print(f'Steps {total_steps}, extra_reward: {extra_rewards:.2f}, rewards: {rewards}')
        if loss is not None:
            wandb.log(loss, step=total_steps)
    #     if total_steps > 1000_000 and 'antmaze' in FLAGS.env_name:
    #         sac_agent.replay_buffer.sav_rep(FLAGS.env_name)
    #         exit()
    # if  'antmaze' in FLAGS.env_name:
    #     sac_agent.replay_buffer.sav_rep(FLAGS.env_name)

# import gym
@torch.no_grad()
def evaluate_policy(env_name, env, agent, high_policy, goal_encode, value_fn, turns = 1, render_num=2, base_observation=None, pred_goal=None):
    size = 200
    total_scores = 0
    steps = 0
    renders = []
    if 'Sawyer' in env_name:
        render_num = 0
    for _ in range(turns):
        s = env.reset()
        done = False
        render = []
        if 'antmaze' in env_name:
            goal = env.wrapped_env.target_goal
            obs_goal = base_observation.copy()
            obs_goal[:2] = goal
        elif 'kitchen' in env_name:
            obs_goal = s[30:]
            obs_goal[:9] = s[:9]
            s = s[:30]
        elif 'maze2d' in env_name:
            goal = env.get_target()
            obs_goal = np.array(goal)
        elif ('FetchReach' in env_name):
            goal = s['desired_goal']
            s = s['observation']
            obs_goal = np.zeros_like(base_observation)
            obs_goal[0:3] = goal
        elif 'FetchPick' in env_name:
            goal = s['desired_goal']
            s = s['observation']
            obs_goal = np.zeros_like(base_observation)
            obs_goal[0:6] = np.concatenate([goal, goal])
        elif 'calvin' in env_name:
            s = s['ob']
            goal = np.array([0.25, 0.15, 0, 0.088, 1, 1])
            obs_goal = base_observation.copy()
            obs_goal[15:21] = goal
        elif 'Sawyer' in env_name:
            goal = s['desired_goal'][0]
            s = s['observation']
            obs_goal = np.zeros_like(base_observation)
            obs_goal[-goal.shape[0]:] = goal

        r = 0
        obs_goal = torch.from_numpy(obs_goal).float().reshape(1, -1).to(device)
        ep_steps = 0
        while not done:
            # Take deterministic actions at test time
            s = s.squeeze()

            with torch.no_grad():
                cur_obs_goal = high_policy(observations=s, goals=obs_goal, temperature=1)
                cur_obs_goal = cur_obs_goal.squeeze(0).cpu().numpy()
            if FLAGS.use_rep:
                cur_obs_goal = cur_obs_goal / np.linalg.norm(cur_obs_goal, axis=-1, keepdims=True) * np.sqrt(cur_obs_goal.shape[-1]).item()
            elif FLAGS.pred_goal:
                cur_obs_goal = cur_obs_goal + s
            
            if (r == 1.0) and 'maze2d' in env_name:
                cur_obs_goal[:2] = goal
                cur_obs_goal[2:] = 0
            cur_obs_goal_rep = cur_obs_goal

            a = agent.select_action(s, cur_obs_goal_rep, deterministic=False)
            if (r == 1.0) and ('Fetch' in env_name or 'Sawyer' in env_name):
                a[:3] = 0

            if len(renders) < render_num:
                if 'kitchen' in env_name:
                    cur_frame = kitchen_render(env, wh=200).transpose(2, 0, 1)
                elif 'calvin' in env_name:
                    cur_frame = env.render(mode='rgb_array').transpose(2, 0, 1).copy()
                else:
                    cur_frame = env.render(mode='rgb_array', width=size, height=size).transpose(2, 0, 1).copy()
                render.append(cur_frame)

            if 'calvin' in env_name:
                s_next, r, done, info = env.step({'ac': np.array(a)})
                s_next = s_next['ob']
                del info['robot_info']
                del info['scene_info']
            else:
                s_next, r, done, info = env.step(a)

            if isinstance(s_next, dict):
                s_next = s_next['observation']
            if 'kitchen' in env_name:
                s_next = s_next[:30]

            r = reward_adapt(r, env_name)
            total_scores += r
            steps+=1
            s = s_next
            ep_steps += 1
        if len(renders) < render_num:
            renders.append(np.array(render))
    return (total_scores/turns), (steps/turns), renders

if __name__ == '__main__':
    app.run(main)












# def xy_to_pixxy(x, y):
                    #     if 'large' in env_name:
                    #         pixx = (x / 36) * (0.93 - 0.07) + 0.07
                    #         pixy = (y / 24) * (0.21 - 0.79) + 0.79
                    #     elif 'ultra' in env_name:
                    #         pixx = (x / 52) * (0.955 - 0.05) + 0.05
                    #         pixy = (y / 36) * (0.19 - 0.81) + 0.81
                    #     return pixx, pixy
                    # x, y = cur_obs_goal_rep[:2]
                    # pixx, pixy = xy_to_pixxy(x, y)
                    # cur_frame[0, int((pixy - 0.02) * size):int((pixy + 0.02) * size), int((pixx - 0.02) * size):int((pixx + 0.02) * size)] = 255
                    # cur_frame[1:3, int((pixy - 0.02) * size):int((pixy + 0.02) * size), int((pixx - 0.02) * size):int((pixx + 0.02) * size)] = 0
