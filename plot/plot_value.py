import os
import time
from datetime import datetime
import gym
import torch
import torch.nn.functional as F
from absl import app, flags
from functools import partial
import numpy as np
from src.agents import hdql as learner

from src import d4rl_utils, d4rl_ant, ant_diagnostics, viz_utils
from src.gc_dataset import GCSDataset

# from rl_m.wandb import setup_wandb, default_wandb_config
# import wandb
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



# config_flags.DEFINE_config_dict('wandb', wandb_config, lock_config=False)
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

    v1, v2 = value_fn(next_observation, goal, encoded=False)
    v = (v1 + v2) / 2

    return v.item()

@torch.no_grad()
def get_reward(state, next_state, goal, value_fn):
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    next_state = torch.from_numpy(next_state).float().unsqueeze(0).to(device)
    goal = torch.from_numpy(goal).float().unsqueeze(0).to(device)

    v1, v2 = value_fn(state, goal)
    _v1, _v2 = value_fn(next_state, goal)
    v = (v1 + v2) / 2
    _v = (_v1 + _v2) / 2

    # scale to [-1, 1]
    advantage = (_v - v).item()
    # if advantage > 1:
    #     advantage = 0
    # # advantage = np.clip(advantage, -1, 1)
    #use sigmoid
    advantage = 2 / (1 + np.exp(-advantage)) - 1

    # v1 = np.linalg.norm(state - goal)
    # v2 = np.linalg.norm(next_state - goal)
    # advantage = v1 - v2
    return advantage

def reward_adapt(reward, env_name):
    # if 'antmaze' in env_name:
    #     reward = reward + 1
    if 'kitchen' in env_name:
        reward = reward
    elif 'maze2d' in env_name:
        reward = reward
    # elif 'calvin' in env_name:
        # reward += 1
    elif 'Fetch' in env_name:
        reward = reward + 1

    # else:
    #     reward = (reward + 1)
    

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
    # elif 'FetchPick' in env_name:
    #     return partial(distance_based_reward, threshold=0.05, dim_range=[3, 6])
    # elif 'FetchPush' in env_name:
    #     return partial(distance_based_reward, threshold=0.05, dim_range=[3, 6])
    elif 'maze2d' in env_name:
        return partial(distance_based_reward, threshold=0.5, dim_range=[0, 2])
    elif 'Hand' in env_name:
        return partial(distance_based_reward, threshold=0.01, dim_range=[0, 15])
    return None

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    # supply_rng(seed)

def main(_):
    # set_seed(FLAGS.seed)
    g_start_time = int(datetime.now().timestamp())

    exp_name = ''
    exp_name += f'sd{FLAGS.seed:03d}_'
    if 'SLURM_JOB_ID' in os.environ:
        exp_name += f's_{os.environ["SLURM_JOB_ID"]}.'
    if 'SLURM_PROCID' in os.environ:
        exp_name += f'{os.environ["SLURM_PROCID"]}.'
    if 'SLURM_RESTART_COUNT' in os.environ:
        exp_name += f'rs_{os.environ["SLURM_RESTART_COUNT"]}.'
    exp_name += f'{g_start_time}'
    # exp_name += f'_{FLAGS.wandb["name"]}'

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
    # FLAGS.wandb['name'] = FLAGS.wandb['exp_descriptor'] = exp_name
    # FLAGS.wandb['group'] = FLAGS.wandb['exp_prefix'] = FLAGS.run_group
    # setup_wandb(params_dict, **FLAGS.wandb)

    # FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, wandb.config.exp_prefix, wandb.config.experiment_id)
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
        viz_env, viz_dataset = d4rl_ant.get_env_and_dataset(env_name)
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
    elif 'Hand' in FLAGS.env_name:
        import gymnasium as gym
        env = gym.make(FLAGS.env_name, render_mode='rgb_array')
        eval_env = gym.make(FLAGS.env_name, render_mode='rgb_array')
        obs_dim = env.observation_space['observation'].shape[0]
        assert FLAGS.load_path is not None
        dataset = d4rl_utils.create_dataset(FLAGS.load_path, obs_dim)
    else:
        env = d4rl_utils.make_env(env_name)
        obs_dim = env.observation_space['observation'].shape[0]
        eval_env = d4rl_utils.make_env(env_name)        
        dataset = d4rl_utils.create_dataset(FLAGS.load_path, obs_dim)


    
    pretrain_dataset = GCSDataset(FLAGS.env_name,dataset, **FLAGS.gcdataset.to_dict())
    # total_steps = FLAGS.pretrain_steps
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


    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import math


    def get_canvas_image(canvas):
        canvas.draw() 
        out_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        out_image = out_image.reshape(canvas.get_width_height()[::-1] + (3,))
        return out_image



    viz_env, viz_dataset = d4rl_ant.get_env_and_dataset(FLAGS.env_name)

    map = viz_env.env.env._maze_map
    maze_size_scaling =  viz_env.env.env._maze_size_scaling
    h, w = len(map), len(map[0])

    bl, tr = viz_env.get_starting_boundary()
    bl, tr = np.array(bl), np.array(tr)
    print(bl, tr)
    step = 0.5
    x_scale = int((tr[0] - bl[0]) / step)
    y_scale = int((tr[1] - bl[1]) / step)
    X = np.linspace(bl[0]+step/2 , tr[0]-step/2 , x_scale,)
    Y = np.linspace(bl[1]+step/2 , tr[1]-step/2 , y_scale,)

    X,Y = np.meshgrid(X,Y)
    states = np.array([X.flatten(), Y.flatten()]).T

    valid_cells = []
    # init map
    coverage_map = np.zeros((y_scale, x_scale))
    value = np.zeros((y_scale, x_scale))
    # 将coverage_map中的值与map中的值对应起来
    for i in range(y_scale):
        for j in range(x_scale):
            # 计算对应的坐标
            x = math.floor((j* w) / x_scale )
            y = math.floor((i * h) / y_scale )
            if map[y][x] in [0, 'r', 'g']:
                coverage_map[i][j] = 1
                valid_cells.append((i, j))

    # 计算所有的点到goal的value
    import gym
    env = gym.make(FLAGS.env_name)
    s = env.reset()
    # goal = env.wrapped_env.target_goal
    # goal is right upper

    if 'umaze' in env_name:
        # upper left
        goal_cell = max(valid_cells, key=lambda x: x[0]-x[1])
    else:
        goal_cell = max(valid_cells, key=lambda x: x[0]+x[1])
    goal = (goal_cell[1] * step + bl[0], goal_cell[0] * step + bl[1])
    goal = [8, 3]
    obs_goal = s.copy()
    obs_goal[:2] = goal[:2]    

    

    # generate sub goal
    # high_policy_fn = partial(agent.sample_high_actions)
    # obs_goal = high_policy_fn(s, obs_goal).squeeze().detach().cpu().numpy()


    medium_goal_i = np.sort([x for x, y in valid_cells], axis=0)[len(valid_cells) // 2]
    medium_goal_j = np.sort([y for x, y in valid_cells], axis=0)[len(valid_cells) // 2]

    print("x, y", medium_goal_i, medium_goal_j)
    goal = (medium_goal_i, medium_goal_j)
    print("is available", coverage_map[medium_goal_i][medium_goal_j])
    obs_goal[:2] = states[goal[0] * x_scale + goal[1]]
    print(obs_goal[:2])

    for i in range(y_scale):
        for j in range(x_scale):
            x_y = states[i * x_scale + j]
            state = s.copy()
            state[:2] = x_y
            value[i][j] = get_guide_v(state, obs_goal, agent.network.value)
    
    # for the same distance, we caculate the std
    
    # 计算value
    def bfs(coverage_map, goal):
        # preprocess goal to grid index
        
        q = [goal]
        visited = set(q)
        value = np.zeros_like(coverage_map)
        while q:
            x, y = q.pop(0)
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy
                if nx < 0 or nx >= y_scale or ny < 0 or ny >= x_scale:
                    continue
                if (nx, ny) in visited:
                    continue
                visited.add((nx, ny))
                if coverage_map[nx][ny] == 1:
                    value[nx][ny] = value[x][y] - 1
                    q.append((nx, ny))

        # normalize
        # return (value - np.min(value)) / (np.max(value) - np.min(value)) * 100
        return value
    # to grid index
    acc_value = bfs(coverage_map, goal)
    # make dict
    value_dict = {}
    for i in range(y_scale):
        for j in range(x_scale):
            if coverage_map[i][j] == 1:
                if acc_value[i][j] in value_dict.keys():
                    value_dict[acc_value[i][j]].append(value[i][j])
                else:
                    value_dict[acc_value[i][j]] = [value[i][j]]
    #caculate std
    std_dict = {}
    for key in value_dict.keys():
        # print(f'key:{key}, value:{value_dict[key]}')
        std_dict[key] = np.std(value_dict[key])
    # print(std_dict.keys(), std_dict.values())
    # exit()
    # show and save std for each distance
    fig, ax = plt.subplots()
    x = -np.array(list(std_dict.keys()))
    y = np.array(list(std_dict.values()))
    # sort by x
    index = np.argsort(x)
    x = x[index]
    y = y[index]
    # smooth y
    y = np.convolve(y, np.ones(10)/10, mode='same')
    x = x[1:47]
    y = y[1:47]
    # 将不同区域的线段设置为不同的颜色
    #[10,20]:yellow, [20,30]:green, [30,40]:orange
    ax.plot(x, y, color='#c9182c', linewidth=4)
    ax.plot(x[10:21], y[10:21], color='#fca311', linewidth=4)
    ax.plot(x[20:31], y[20:31], color='#006d77', linewidth=4)
    ax.plot(x[30:41], y[30:41], color='#ffb3ae', linewidth=4)
    
    # set tick and tick font size
    ax.set_xticks(np.arange(0, 46, 15))
    ax.set_yticks(np.arange(0, 9, 2))
    ax.tick_params(labelsize=18)
    ax.set_xlabel('Distance between $s$ and $g$', fontsize=24)
    ax.set_ylabel('Std of $ V(s,g) $', fontsize=24)
    plt.tight_layout()
    plt.show()
    plt.savefig('coverage/std.pdf')

    std_mesh = np.zeros((y_scale, x_scale))
    for i in range(y_scale):
        for j in range(x_scale):
            if coverage_map[i][j] == 1:
                std_mesh[i][j] = std_dict[acc_value[i][j]]
    # show std in map, use mesh
    fig, ax = plt.subplots()
    # fontsize
    plt.rcParams.update({'font.size': 24})
    mesh = ax.pcolormesh(X, Y, value, cmap='coolwarm', alpha=0.5)
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('left', size='5%', pad=0.2)
    # label: $V(s, s_g)$ using LaTex, set label to left
    cbar = fig.colorbar(mesh, cax=cax, orientation='vertical', label='$V(s, g)$')
    # cbar.set_label('$V(s, g)$', rotation=270, labelpad=20)
    cbar.ax.yaxis.set_label_position('left')
    cbar.ax.yaxis.tick_left()
    viz_env.draw(ax)
    ax.set_aspect('equal')


    acc_value[coverage_map == 0] = np.nan
    ax.contour(X, Y, -acc_value, levels=[10,19,20,29,30,39], colors=['#fca311', '#fca311', '#006d77', '#006d77', '#ffb3ae', '#ffb3ae'], linewidths=2, linestyles='dashed', alpha=1)
    plt.tight_layout()
    plt.show()
    plt.savefig('coverage/std_map.pdf')
    exit()

    # Compute the gradient of value, i want gradient of coverage_map == 0 is 0
    value[coverage_map == 0] = np.nan
    if 'umaze' in env_name:
        goal_x_y = max(valid_cells, key=lambda x: x[0]-x[1])
    else:
        goal_x_y = max(valid_cells, key=lambda x: x[0]+x[1])
    # goal_pos = states[goal_x_y]

    # Create a grid of x and y coordinates
    # y, x = np.mgrid[0:value.shape[0], 0:value.shape[1]]

    # Create a quiver plot
    dy, dx = np.gradient(value)
    # Replace np.nan with 0 in the gradients
    dy = np.nan_to_num(dy)
    dx = np.nan_to_num(dx)
    clip_value = np.percentile(np.sqrt(dx**2 + dy**2), 90)
    index = np.sqrt(dx**2 + dy**2) < clip_value
    dx[index] /= clip_value
    dy[index] /= clip_value

    # other norm to 1
    dx[~index] /= np.sqrt(dx[~index]**2 + dy[~index]**2)
    dy[~index] /= np.sqrt(dx[~index]**2 + dy[~index]**2)

    # Compute the 75th percentile of the gradients
    

    # Clip the gradients
    

    # 计算value[i][j]的梯度
    # dx = np.zeros_like(value)
    # dy = np.zeros_like(value)

    # max_norm = 0
    # for i in range(y_scale):
    #     for j in range(x_scale):
    #         if coverage_map[i][j] != 1:
    #             continue

    #         if coverage_map[i+1][j] == 1 and coverage_map[i-1][j] == 1:
    #             dx[i][j] = (value[i+1][j] - value[i-1][j]) / 2
    #         else:
    #             dx[i][j] = 0
            
    #         if coverage_map[i][j+1] == 1 and coverage_map[i][j-1] == 1:
    #             dy[i][j] = (value[i][j+1] - value[i][j-1]) / 2
    #         else:
    #             dy[i][j] = 0

    # # normalize 为单位向量
    # dx = dx / max_norm
    # dy = dy / max_norm

    # smooth the quiver to make it more readable
    # def gaussian_filter(field, sigma=1):
    #     from scipy.ndimage import gaussian_filter
    #     return gaussian_filter(field, sigma=sigma)
    # dx = gaussian_filter(dx, sigma=1)
    # dy = gaussian_filter(dy, sigma=1)


    # plot quiver
    fig, ax = plt.subplots()
    # the larger the scale, the smaller the arrow
    ax.quiver(X, Y, dx, dy, scale=30)

    # plot goal
    # ax.plot(goal_pos[0], goal_pos[1], 'ro', size=10)

    viz_env.draw(ax)
    ax.set_aspect('equal')
    plt.show()
    plt.savefig('coverage/quiver.png')

if __name__ == '__main__':
    app.run(main)