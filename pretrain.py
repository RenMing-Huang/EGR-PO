import os
import time
from datetime import datetime

import torch

from absl import app, flags
import numpy as np

import gzip

import tqdm
from src.agents import hdql as learner
from src import d4rl_utils, d4rl_ant, ant_diagnostics, viz_utils
from src.gc_dataset import GCSDataset

from rl_m.wandb import setup_wandb, default_wandb_config
import wandb
from rl_m.evaluation import EpisodeMonitor

from ml_collections import config_flags
import pickle

from src.utils import CsvLogger
from src.agents.hdql import device



def view_data_distribution(viz_env, ds):
    vobs = ds["observations"][..., :2]
    return d4rl_ant.plot_points(viz_env, vobs[:, 0], vobs[:, 1])

FLAGS = flags.FLAGS
flags.DEFINE_string('env_name', 'antmaze-large-diverse-v2', '')
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

flags.DEFINE_string('algo_name', 'egr_po', '')  # Not used, only for logging

flags.DEFINE_string('load_path', None, '')
flags.DEFINE_string('pred_goal', None, '')
flags.DEFINE_string('dataset_type','normal', 'normal, mini, incomplete')


wandb_config = default_wandb_config()
wandb_config.update({
    'project': 'egr-po',
    'group': 'Debug',
    'name': '{env_name}',
})

config_flags.DEFINE_config_dict('wandb', wandb_config, lock_config=False)
config_flags.DEFINE_config_dict('config', learner.get_default_config(), lock_config=False)

gcdataset_config = GCSDataset.get_default_config()
config_flags.DEFINE_config_dict('gcdataset', gcdataset_config, lock_config=False)



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



@torch.no_grad()
def get_traj_v(agent, trajectory):
    observations = trajectory['observations']
    n = observations.shape[0]
    obs_t = torch.from_numpy(observations).float().to(device)
    # Compute all (n × n) state-goal value pairs in a single batched forward pass.
    states = obs_t.unsqueeze(1).expand(n, n, -1).reshape(n * n, -1)
    goals = obs_t.unsqueeze(0).expand(n, n, -1).reshape(n * n, -1)
    v1, v2 = agent.network.value(states, goals)
    all_values = ((v1 + v2) / 2).reshape(n, n).cpu().numpy()
    return {
        'dist_to_beginning': all_values[:, 0],
        'dist_to_end': all_values[:, -1],
        'dist_to_middle': all_values[:, n // 2],
    }

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def main(_):
    g_start_time = int(datetime.now().timestamp())

    exp_name = f'sd{FLAGS.seed:03d}_'
    if 'SLURM_JOB_ID' in os.environ:
        exp_name += f's_{os.environ["SLURM_JOB_ID"]}.'
    if 'SLURM_PROCID' in os.environ:
        exp_name += f'{os.environ["SLURM_PROCID"]}.'
    if 'SLURM_RESTART_COUNT' in os.environ:
        exp_name += f'rs_{os.environ["SLURM_RESTART_COUNT"]}.'
    # exp_name += f'{g_start_time}'
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
    FLAGS.config['goal_dim'] = FLAGS.goal_dim
    FLAGS.config['pred_goal'] = FLAGS.pred_goal
    FLAGS.config['dataset_type'] = FLAGS.dataset_type

    # Create wandb logger
    params_dict = {**FLAGS.gcdataset.to_dict(), **FLAGS.config.to_dict()}
    FLAGS.wandb['name'] = FLAGS.wandb['exp_descriptor'] = exp_name
    FLAGS.wandb['group'] = FLAGS.wandb['exp_prefix'] = FLAGS.run_group
    setup_wandb(params_dict, **FLAGS.wandb)

    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, wandb.config.exp_prefix, wandb.config.experiment_id)
    os.makedirs(FLAGS.save_dir, exist_ok=True)

    goal_info = None
    discrete = False
    if 'antmaze' in FLAGS.env_name:
        env_name = FLAGS.env_name
        if 'ultra' in FLAGS.env_name:
            import d4rl_ext
            import gym
            env = gym.make(env_name)
            env = EpisodeMonitor(env)
        else:
            env = d4rl_utils.make_env(env_name)

        dataset = d4rl_utils.get_dataset(env, FLAGS.env_name, dataset_type = FLAGS.dataset_type)
        dataset = dataset.copy({'rewards': dataset['rewards'] - 1.0})
        # show dataset size
        print(f"Use Dataset: {FLAGS.dataset_type}")
        print(f"Dataset size: {len(dataset['observations'])}")

        

        env.render(mode='rgb_array', width=200, height=200)
        if 'large' in FLAGS.env_name:
            env.viewer.cam.lookat[0] = 18
            env.viewer.cam.lookat[1] = 12
            env.viewer.cam.distance = 50
            env.viewer.cam.elevation = -90

            # viz_env, viz_dataset = d4rl_ant.get_env_and_dataset(env_name)
            # viz = ant_diagnostics.Visualizer(env_name, viz_env, viz_dataset, discount=FLAGS.discount)

        elif 'ultra' in FLAGS.env_name:
            env.viewer.cam.lookat[0] = 26
            env.viewer.cam.lookat[1] = 18
            env.viewer.cam.distance = 70
            env.viewer.cam.elevation = -90
        else:
            env.viewer.cam.lookat[0] = 18
            env.viewer.cam.lookat[1] = 12
            env.viewer.cam.distance = 50
            env.viewer.cam.elevation = -90
    elif 'kitchen' in FLAGS.env_name:
        env = d4rl_utils.make_env(FLAGS.env_name)
        dataset = d4rl_utils.get_dataset(env, FLAGS.env_name, filter_terminals=True)
        dataset = dataset.copy({'observations': dataset['observations'][:, :30], 'next_observations': dataset['next_observations'][:, :30],
                                'achieved_goals': dataset['observations'][:, :30], 'next_achieved_goals': dataset['next_observations'][:, :30]})
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
    elif 'procgen' in FLAGS.env_name:
        from src.envs.procgen_env import ProcgenWrappedEnv, get_procgen_dataset
        import matplotlib

        matplotlib.use('Agg')

        n_processes = 1
        env_name = 'maze'
        env = ProcgenWrappedEnv(n_processes, env_name, 1, 1)

        if FLAGS.env_name == 'procgen-500':
            dataset = get_procgen_dataset('data/procgen/level500.npz', state_based=('state' in FLAGS.env_name))
            min_level, max_level = 0, 499
        elif FLAGS.env_name == 'procgen-1000':
            dataset = get_procgen_dataset('data/procgen/level1000.npz', state_based=('state' in FLAGS.env_name))
            min_level, max_level = 0, 999
        else:
            raise NotImplementedError

        # Test on large levels having >=20 border states
        large_levels = [12, 34, 35, 55, 96, 109, 129, 140, 143, 163, 176, 204, 234, 338, 344, 369, 370, 374, 410, 430, 468, 470, 476, 491] + [5034, 5046, 5052, 5080, 5082, 5142, 5244, 5245, 5268, 5272, 5283, 5335, 5342, 5366, 5375, 5413, 5430, 5474, 5491]
        goal_infos = []
        goal_infos.append({'eval_level': [level for level in large_levels if min_level <= level <= max_level], 'eval_level_name': 'train'})
        goal_infos.append({'eval_level': [level for level in large_levels if level > max_level], 'eval_level_name': 'test'})

        dones_float = 1.0 - dataset['masks']
        dones_float[-1] = 1.0
        dataset = dataset.copy({
            'dones_float': dones_float
        })

        discrete = True
        example_action = np.max(dataset['actions'], keepdims=True)
    elif 'maze2d' in FLAGS.env_name:
        env = d4rl_utils.make_env(FLAGS.env_name)
        dataset = d4rl_utils.get_dataset(env, FLAGS.env_name)
        dataset = dataset.copy({'rewards': dataset['rewards'] - 1.0})
        
    elif 'Fetch' in FLAGS.env_name:
        env = d4rl_utils.make_env(FLAGS.env_name)
        obs_dim = env.observation_space['observation'].shape[0]
        assert FLAGS.load_path is not None
        if 'FetchPush' in FLAGS.env_name:
            dataset = d4rl_utils.create_dataset(FLAGS.load_path, obs_dim, goal_dim=[3,6])
        else:
            dataset = d4rl_utils.create_dataset(FLAGS.load_path, obs_dim)
    
    elif 'SawyerReach' in FLAGS.env_name:
        from wgcsl.common.env_util import get_env_type, build_env, get_game_envs
        _game_envs = get_game_envs()
        env_type, env_id = get_env_type('SawyerReach', _game_envs)
        print('env_type: {}'.format(env_type))
        env = build_env('SawyerReach', _game_envs)
        obs_dim = env.observation_space['observation'].shape[0]
        dataset = d4rl_utils.create_dataset(FLAGS.load_path, obs_dim)
    elif 'SawyerDoor' in FLAGS.env_name:
        from wgcsl.common.env_util import get_env_type, build_env, get_game_envs
        _game_envs = get_game_envs()
        env_type, env_id = get_env_type('SawyerDoor', _game_envs)
        print('env_type: {}'.format(env_type))
        env = build_env('SawyerDoor', _game_envs)
        obs_dim = env.observation_space['observation'].shape[0]
        dataset = d4rl_utils.create_dataset(FLAGS.load_path, obs_dim, goal_dim=[3,4])


    pretrain_dataset = GCSDataset(FLAGS.env_name, dataset, **FLAGS.gcdataset.to_dict())
    total_steps = FLAGS.pretrain_steps
    example_batch = dataset.sample(1)
    agent = learner.create_learner(FLAGS.seed,
                                   example_batch['observations'],
                                   example_batch['actions'] if not discrete else example_action,
                                   visual=FLAGS.visual,
                                   encoder=FLAGS.encoder,
                                   discrete=discrete,
                                   use_layer_norm=FLAGS.use_layer_norm,
                                   rep_type=FLAGS.rep_type,
                                   **FLAGS.config)

    # For debugging metrics
    if 'antmaze' in FLAGS.env_name:
        example_trajectory = pretrain_dataset.sample(50, indx=np.arange(1000, 1050))
    elif 'kitchen' in FLAGS.env_name:
        example_trajectory = pretrain_dataset.sample(50, indx=np.arange(0, 50))
    elif 'calvin' in FLAGS.env_name:
        example_trajectory = pretrain_dataset.sample(50, indx=np.arange(0, 50))
    elif 'procgen-500' in FLAGS.env_name:
        example_trajectory = pretrain_dataset.sample(50, indx=np.arange(5000, 5050))
    elif 'procgen-1000' in FLAGS.env_name:
        example_trajectory = pretrain_dataset.sample(50, indx=np.arange(5000, 5050))
    else:
        example_trajectory = pretrain_dataset.sample(50, indx=np.arange(0, 50))

    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()
    for i in tqdm.tqdm(range(1, total_steps + 1),
                       smoothing=0.1,
                       dynamic_ncols=True):
        pretrain_batch = pretrain_dataset.sample(FLAGS.batch_size)
        agent, loss, update_info = agent.pretrain_update(pretrain_batch, actor_update=False)
        if i == 1 or i % FLAGS.log_interval == 0:
            debug_statistics = get_debug_statistics(agent, pretrain_batch)
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            train_metrics.update({f'pretraining/debug/{k}': v for k, v in debug_statistics.items()})
            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = (time.time() - first_time)
            last_time = time.time()
            wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        if i % FLAGS.save_interval == 0:
            save_dict = dict(
                agent=agent.network.state_dict(),
                config=FLAGS.config.to_dict()
            )

            fname = os.path.join(FLAGS.save_dir, f'params_{i}.pkl')
            print(f'Saving to {fname}')
            with open(fname, "wb") as f:
                pickle.dump(save_dict, f)
    train_logger.close()
    eval_logger.close()

def format_print(_dict):
    print('============================')
    for k, v in _dict.items():
        print(f'{k}: {v:.4f}')
    print('============================')

if __name__ == '__main__':
    app.run(main)
