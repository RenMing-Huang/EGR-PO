
from src import d4rl_ant, d4rl_utils
import d4rl
import gym
import numpy as np
import math
import d4rl_ext
import seaborn as sns
from collections import defaultdict
import matplotlib

# matplotlib.rcParams['font.family'] = 'Times New Roman'
def add_to(dict_of_lists, single_dict):
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)
for env in ['antmaze-umaze-v2', 'antmaze-large-play-v2','antmaze-medium-play-v2']: #'antmaze-medium-play-v2', 'antmaze-ultra-play-v0','antmaze-large-play-v2'
    env_name = env

    viz_env, viz_dataset = d4rl_ant.get_env_and_dataset(env_name)

    map = viz_env.env.env._maze_map
    maze_size_scaling =  viz_env.env.env._maze_size_scaling
    # viz_env = viz_env.env.env
    h, w = len(map), len(map[0])
    print("h, w", h, w)


    print(maze_size_scaling)

    valid_cells = []


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
    if 'umaze' in env_name:
        # upper left
        goal = max(valid_cells, key=lambda x: x[0]-x[1])
    else:
        # upper right
        goal = max(valid_cells, key=lambda x: x[0]+x[1])
    # value
    def bfs(coverage_map, goal):
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
        # print(value.min(), value.max())
        return (value - np.min(value)) / (np.max(value) - np.min(value)) * 100

    value = bfs(coverage_map, goal)


    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from mpl_toolkits.axes_grid1 import make_axes_locatable


    def get_canvas_image(canvas):
        canvas.draw() 
        out_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        out_image = out_image.reshape(canvas.get_width_height()[::-1] + (3,))
        return out_image
    wight = w
    height = h
    ratio = wight / height
    figsize=(6.4 * ratio, 6.4)
    fig = plt.figure(figsize=figsize, tight_layout=False)
    # set font size
    plt.rcParams.update({'font.size': 24})
    canvas = FigureCanvas(fig)

    x, y = states[:, 0], states[:, 1]
    x = x.reshape(y_scale,x_scale)
    y = y.reshape(y_scale,x_scale)
    ax = plt.gca()

    # goal to xy
    # goal = np.array(goal)
    # env = gym.make(env_name)
    # s = env.reset()
    # goal = env.wrapped_env.target_goal
    # goal = np.array(goal)
    # # goal[0] = (goal[0] - bl[0]) / (tr[0] - bl[0]) * x_scale
    # # goal[1] = (goal[1] - bl[1]) / (tr[1] - bl[1]) * y_scale

    # goal = (goal - bl) / (tr - bl) * np.array([x_scale, y_scale])
    # goal = goal.astype(int)

    # exit()
    # value[goal[1], goal[0]] = 10
    # value[s[1], s[0]] = -200

    mesh = ax.pcolormesh(x, y, value, cmap='coolwarm', alpha=0.5)
    # set title
    ax.set_title('antmaze-large', fontsize=30)
    viz_env.draw(ax)
    divider = make_axes_locatable(ax)
    # if 'ultra' in env_name:
    #     cax = divider.append_axes('right', size='5%', pad=0.2)
    #     fig.colorbar(mesh, cax=cax, orientation='vertical', label="Importance $\\left(\\% \\right)$")


    image = get_canvas_image(canvas)
    
    plt.close(fig)

    # save image
    import os
    os.makedirs('coverage', exist_ok=True)
    
    plt.imsave(f'coverage/coverage_map_{env_name}.pdf', image)

    # save colorbar
    fig, ax = plt.subplots(figsize=(3, 7))
    # fig.subplots_adjust(bottom=0.5)
    
    # Create a colorbar
    import matplotlib as mpl
    plt.rcParams.update({'font.size': 30})
    cmap = mpl.cm.coolwarm
    norm = mpl.colors.Normalize(vmin=0, vmax=100)
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='vertical', label="Importance $\\left(\\% \\right)$", alpha=0.5)
    # set color bar more thin
    fig.tight_layout()
    
    # Save the colorbar
    plt.savefig('coverage/colorbar.pdf')
    plt.close(fig)

    buffer_paths = {
        'antmaze-large-play-v2': {
            'Online': f'exp_data/online-s0/buffers/buffer_antmaze-large-play-v2.npz',
            'RND': f'exp_data/online_rnd-s0/buffers/buffer_antmaze-large-play-v2.npz',
            'ExPLORe': f'exp_data/ours-s0/buffers/buffer_antmaze-large-play-v2.npz',
            'Ours':'buffers/antmaze-large-play-v2.npy'
        },
        'antmaze-medium-play-v2':
        {
            'Online': f'exp_data/online-s0/buffers/buffer_antmaze-medium-play-v2.npz',
            'RND': f'exp_data/online_rnd-s0/buffers/buffer_antmaze-medium-play-v2.npz',
            'ExPLORe': f'exp_data/ours-s0/buffers/buffer_antmaze-medium-play-v2.npz',
            'Ours':'buffers/antmaze-medium-play-v2.npy'
        },
        'antmaze-umaze-v2':{
            'Online': f'exp_data/online-s0/buffers/buffer_antmaze-umaze-v2.npz',
            'RND': f'exp_data/online_rnd-s0/buffers/buffer_antmaze-umaze-v2.npz',
            'ExPLORe': f'exp_data/ours-s0/buffers/buffer_antmaze-umaze-v2.npz',
            'Ours':'buffers/antmaze-umaze-v2.npz'
            },
        'antmaze-ultra-play-v0':{
            'Online': f'exp_data/online-s0/buffers/buffer_antmaze-ultra-play-v0.npz',
            'RND': f'exp_data/online_rnd-s0/buffers/buffer_antmaze-ultra-play-v0.npz',
            'ExPLORe': f'exp_data/ours-s0/buffers/buffer_antmaze-ultra-play-v0.npz',
            'Ours': 'buffers/antmaze-ultra-diverse-v0.npy'
        }
    }
    color_map = {
        'Ours': '#c9182c',
        'ExPLORe': '#006d77',
        'RND': '#fca311',
        'Online':'#118ab2',
    }

    def load_buffer(name, env):
        if name=='Ours':
            replay_buffer_path = buffer_paths[env][name]
            replay_buffer = np.load(replay_buffer_path, allow_pickle=True).item()
            if 'size' not in replay_buffer.keys():
                replay_buffer['observation'] = replay_buffer['obs']
                replay_buffer['size'] = len(replay_buffer['observation'])
                replay_buffer['length'] = replay_buffer['episode_len']
            #flat observation
            size = replay_buffer['size']
            observations = []
            total_length = 0
            for i in range(size):
                length = replay_buffer['length'][i].astype(int)
                total_length += length
                obs = replay_buffer['observation'][i][:length]
                observations.append(obs)
                if total_length >= 1000000:
                    break
            observations = np.concatenate(observations, axis=0)

        else:
            replay_buffer_path = buffer_paths[env][name]
            with open(replay_buffer_path, 'rb') as f:
                observations = np.load(f)["observations"]
                
        print(name, len(observations))
        return observations[:1000000]

    all_buffers = {}

    for name in ['Online', 'RND', 'ExPLORe', 'Ours']:
        all_buffers[name] = load_buffer(name, env)

    # value作为权重，计算带权重coverge
    # 首先处理value
    value = value.flatten()
    # value = np.exp(value)

    weighted_coverage = 0
    fig = plt.figure()
    # font size
    plt.rcParams.update({'font.size': 24})

    for i, (name, observations) in enumerate(all_buffers.items()):
        weighted_coverage = 0
        coverage_list = []
        for i in range(observations.shape[0]):
            x, y = observations[i][:2]
            x = (x - bl[0]) / (tr[0] - bl[0]) * x_scale
            y = (y - bl[1]) / (tr[1] - bl[1]) * y_scale
            x, y = int(x), int(y)
            weighted_coverage += value[y * x_scale + x]

            if (i+1) % 1000 == 0:
                coverage_list.append(weighted_coverage / i )

        #plot coverage
        sns.lineplot(x=np.linspace(0, len(coverage_list)*1000, len(coverage_list)), y=coverage_list, color=color_map[name], label = name, linewidth=3)
        plt.xlabel('Environment Steps', fontsize=24)
        plt.title(env, fontsize=24)
        plt.ylim([0, 100])
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    if 'umaze' in env_name:
        # upper left
        # plt.ylabel('Weighted Coverage', fontsize=24)
        plt.legend(loc='upper left', fontsize=16)
    else:
        # remove y label
        plt.ylabel('')
        # remove legend
        plt.legend().remove()
    plt.tight_layout()
    plt.savefig(f'coverage/coverage_{env_name}.pdf')

    # #plot coverage
    # fig = plt.figure()
    # plt.plot(coverage_list)
    # plt.xlabel('steps')
    # plt.ylabel('weighted coverage')
    # plt.title('weighted coverage')
    # plt.savefig('coverage/coverage.png')


