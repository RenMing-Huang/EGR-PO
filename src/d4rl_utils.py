import d4rl
import gym
import numpy as np
import os
from rl_m.dataset import Dataset
from rl_m.evaluation import EpisodeMonitor

def create_dataset(load_path, obs_dim, goal_dim=None):
    keys = ['state', 'action', 'reward', 'not_done', 'next_state', ]
    path_list = [os.path.join(load_path, f'{key}.npy') for key in keys]
    dataset = dict()
    for key, path in zip(keys, path_list):
        dataset[key] = np.load(path).squeeze()
        print('key: ', key, ' shape: ', dataset[key].shape)
        if 'state' in key or 'next_state' in key:
            # For Fetch 3:6 is the achieved goal == -3:0
            dataset[key] = dataset[key][:,:obs_dim]

    dones_float = 1.0 - dataset['not_done']

    for i in range(len(dones_float)-1):
        if dataset['reward'][i] < dataset['reward'][i+1]:
            dones_float[i] = 1

        # if dataset['reward'][i] == 1:
        #     dataset['state'][i+1] = dataset['state'][i]
        #     dataset['next_state'][i] = dataset['state'][i]

    # mask = dataset['reward'][:-1] < dataset['reward'][1:]
    # dones_float[:-1][mask] = 1

    # reward_mask = dataset['reward'] == 1
    # dataset['state'][1:][reward_mask[:-1]] = dataset['state'][:-1][reward_mask[:-1]]
    # dataset['next_state'][:-1][reward_mask[:-1]] = dataset['state'][:-1][reward_mask[:-1]]


    dones_float[-1] = 1


    if goal_dim is not None:
        achieved_goals = dataset['state'][:,goal_dim[0]:goal_dim[1]]
        next_achieved_goals= dataset['next_state'][:,goal_dim[0]:goal_dim[1]]
    else:
        achieved_goals = dataset['state']
        next_achieved_goals= dataset['next_state']

    return Dataset.create(
            observations=dataset['state'].astype(np.float32),
            actions= dataset['action'].astype(np.float32),
            rewards= dataset['reward'].astype(np.float32),
            masks= dataset['not_done'].astype(np.float32),
            dones_float= 1.0 - dataset['not_done'].astype(np.float32),
            next_observations=dataset['next_state'].astype(np.float32),
            achieved_goals=achieved_goals.astype(np.float32),
            next_achieved_goals=next_achieved_goals.astype(np.float32),
        )


def make_env(env_name: str):
    env = gym.make(env_name)
    env = EpisodeMonitor(env)
    return env

def get_dataset(env: gym.Env,
                env_name: str,
                clip_to_eps: bool = True,
                eps: float = 1e-5,
                dataset=None,
                filter_terminals=False,
                obs_dtype=np.float32,
                dataset_type = 'normal',
                ):
        if dataset is None:
            dataset = d4rl.qlearning_dataset(env)

        if clip_to_eps:
            lim = 1 - eps
            dataset['actions'] = np.clip(dataset['actions'], -lim, lim)

        dataset['terminals'][-1] = 1
        if filter_terminals:
            # drop terminal transitions
            non_last_idx = np.nonzero(~dataset['terminals'])[0]
            last_idx = np.nonzero(dataset['terminals'])[0]
            penult_idx = last_idx - 1
            new_dataset = dict()
            for k, v in dataset.items():
                if k == 'terminals':
                    v[penult_idx] = 1
                new_dataset[k] = v[non_last_idx]
            dataset = new_dataset
        
        if 'antmaze' in env_name:
            # antmaze: terminals are incorrect for GCRL
            
            dones_float = np.zeros_like(dataset['rewards'])
            dataset['terminals'][:] = 0.
            for i in range(len(dones_float) - 1):
                if np.linalg.norm(dataset['observations'][i + 1] - dataset['next_observations'][i]) > 1e-6:
                    dones_float[i] = 1
                else:
                    dones_float[i] = 0
            dones_float[-1] = 1
            if dataset_type =='mini':
                # sample 1% full trajectories for training
                # first organize the data into full trajectories
                full_trajectories = []
                trajectory = []
                for i in range(len(dones_float)):
                    trajectory.append(i)
                    if dones_float[i] == 1:
                        full_trajectories.append(trajectory)
                        trajectory = []
                # sample 10% of the full trajectories
                np.random.seed(0)
                sampled_trajectories = np.random.choice(len(full_trajectories), len(full_trajectories)//10, replace=False)
                sampled_indices = []
                for i in sampled_trajectories:
                    sampled_indices.extend(full_trajectories[i])
                # sample the data

                for key in dataset.keys():
                    dataset[key] = dataset[key][sampled_indices]
                dones_float = dones_float[sampled_indices]

            if dataset_type == 'incomplete':
                # 将轨迹随机切分为多条轨迹，分为3条
                trajectory = []
                for i in range(len(dones_float)):
                    trajectory.append(i)
                    if dones_float[i] == 1:
                        sample_end = np.random.choice(trajectory[:-1], 2, replace=False)
                        # sample_end = np.sort(sample_end)
                        dones_float[sample_end] = 1
                        trajectory = []
            if dataset_type == 'unsuccess':
                # drop the successful trajectories
                # success: x > 25 and y > 20
                mask_range = [[0,0,3,3], [25,20,35,30], [18,10,21,12]] # 0:mask begin, 1:mask end, 2:mask mid
                mask_type = 1
                mask = mask_range[mask_type]
                full_trajectories = []
                trajectory = []
                for i in range(len(dones_float)):
                    trajectory.append(i)
                    if dones_float[i] == 1:
                        obs = dataset['observations'][trajectory]
                        # get the index of the obs that x > mask[0] and x < mask[2]
                        index = np.where((obs[:, 0] > mask[0]) & (obs[:, 0] < mask[2]))[0]
                        success = np.where((obs[index, 1] > mask[1]) & (obs[index, 1] < mask[3]))[0]
                        if not np.any(success):
                            full_trajectories.extend(trajectory)
                        trajectory = []

                for key in dataset.keys():
                    dataset[key] = dataset[key][full_trajectories]
                achieved_goals = achieved_goals[full_trajectories]
                next_achieved_goals = next_achieved_goals[full_trajectories]
                dones_float = dones_float[full_trajectories]


        elif 'maze2d' in env_name:
            dones_float = dataset['terminals'].copy()
            dataset['terminals'][:] = 0.
            for i in range(len(dones_float)-1):
                if dataset['rewards'][i] == 0 and dataset['rewards'][i+1] == 1:
                    dones_float[i] = 1

            dones_float[-1] = 1
            achieved_goals = dataset['observations'][:, :2]
            next_achieved_goals = dataset['next_observations'][:, :2]
        else:
            dones_float = dataset['terminals'].copy()

        observations = dataset['observations'].astype(obs_dtype)
        next_observations = dataset['next_observations'].astype(obs_dtype)

        achieved_goals = dataset['observations']
        next_achieved_goals= dataset['next_observations']

        return Dataset.create(
            observations=observations,
            actions=dataset['actions'].astype(np.float32),
            rewards=dataset['rewards'].astype(np.float32),
            masks=1.0 - dones_float.astype(np.float32),
            dones_float=dones_float.astype(np.float32),
            next_observations=next_observations,
            achieved_goals=achieved_goals,
            next_achieved_goals=next_achieved_goals,
        )

def get_normalization(dataset):
        returns = []
        ret = 0
        for r, term in zip(dataset['rewards'], dataset['dones_float']):
            ret += r
            if term:
                returns.append(ret)
                ret = 0
        return (max(returns) - min(returns)) / 1000

def normalize_dataset(env_name, dataset):
    if 'antmaze' in env_name:
         return  dataset.copy({'rewards': dataset['rewards']- 1.0})
    else:
        normalizing_factor = get_normalization(dataset)
        dataset = dataset.copy({'rewards': dataset['rewards'] / normalizing_factor})
        return dataset
    

import numpy as np
import scipy.interpolate as interpolate
import pdb

POINTMASS_KEYS = ['observations', 'actions', 'next_observations', 'deltas']

#-----------------------------------------------------------------------------#
#--------------------------- multi-field normalizer --------------------------#
#-----------------------------------------------------------------------------#

class DatasetNormalizer:

    def __init__(self, dataset, normalizer, path_lengths=None):
        dataset = flatten(dataset, path_lengths)

        self.observation_dim = dataset['observations'].shape[1]
        self.action_dim = dataset['actions'].shape[1]

        if type(normalizer) == str:
            normalizer = eval(normalizer)

        self.normalizers = {}
        for key, val in dataset.items():
            try:
                self.normalizers[key] = normalizer(val)
            except:
                print(f'[ utils/normalization ] Skipping {key} | {normalizer}')

    def __repr__(self):
        string = ''
        for key, normalizer in self.normalizers.items():
            string += f'{key}: {normalizer}]\n'
        return string

    def __call__(self, *args, **kwargs):
        return self.normalize(*args, **kwargs)

    def normalize(self, x, key):
        return self.normalizers[key].normalize(x)

    def unnormalize(self, x, key):
        return self.normalizers[key].unnormalize(x)

    def get_field_normalizers(self):
        return self.normalizers

def flatten(dataset, path_lengths):
    '''
        flattens dataset of { key: [ n_episodes x max_path_lenth x dim ] }
            to { key : [ (n_episodes * sum(path_lengths)) x dim ]}
    '''
    flattened = {}
    for key, xs in dataset.items():
        assert len(xs) == len(path_lengths)
        flattened[key] = np.concatenate([
            x[:length]
            for x, length in zip(xs, path_lengths)
        ], axis=0)
    return flattened

#-----------------------------------------------------------------------------#
#------------------------------- @TODO: remove? ------------------------------#
#-----------------------------------------------------------------------------#

class PointMassDatasetNormalizer(DatasetNormalizer):

    def __init__(self, preprocess_fns, dataset, normalizer, keys=POINTMASS_KEYS):

        reshaped = {}
        for key, val in dataset.items():
            dim = val.shape[-1]
            reshaped[key] = val.reshape(-1, dim)

        self.observation_dim = reshaped['observations'].shape[1]
        self.action_dim = reshaped['actions'].shape[1]

        if type(normalizer) == str:
            normalizer = eval(normalizer)

        self.normalizers = {
            key: normalizer(reshaped[key])
            for key in keys
        }

#-----------------------------------------------------------------------------#
#-------------------------- single-field normalizers -------------------------#
#-----------------------------------------------------------------------------#

class Normalizer:
    '''
        parent class, subclass by defining the `normalize` and `unnormalize` methods
    '''

    def __init__(self, X):
        self.X = X.astype(np.float32)
        self.mins = X.min(axis=0)
        self.maxs = X.max(axis=0)

    def __repr__(self):
        return (
            f'''[ Normalizer ] dim: {self.mins.size}\n    -: '''
            f'''{np.round(self.mins, 2)}\n    +: {np.round(self.maxs, 2)}\n'''
        )

    def __call__(self, x):
        return self.normalize(x)

    def normalize(self, *args, **kwargs):
        raise NotImplementedError()

    def unnormalize(self, *args, **kwargs):
        raise NotImplementedError()


class DebugNormalizer(Normalizer):
    '''
        identity function
    '''

    def normalize(self, x, *args, **kwargs):
        return x

    def unnormalize(self, x, *args, **kwargs):
        return x


class GaussianNormalizer(Normalizer):
    '''
        normalizes to zero mean and unit variance
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.means = self.X.mean(axis=0)
        self.stds = self.X.std(axis=0)
        self.z = 1

    def __repr__(self):
        return (
            f'''[ Normalizer ] dim: {self.mins.size}\n    '''
            f'''means: {np.round(self.means, 2)}\n    '''
            f'''stds: {np.round(self.z * self.stds, 2)}\n'''
        )

    def normalize(self, x):
        return (x - self.means) / self.stds

    def unnormalize(self, x):
        return x * self.stds + self.means


class LimitsNormalizer(Normalizer):
    '''
        maps [ xmin, xmax ] to [ -1, 1 ]
    '''

    def normalize(self, x):
        ## [ 0, 1 ]
        x = (x - self.mins) / (self.maxs - self.mins)
        ## [ -1, 1 ]
        x = 2 * x - 1
        return x

    def unnormalize(self, x, eps=1e-4):
        '''
            x : [ -1, 1 ]
        '''
        if x.max() > 1 + eps or x.min() < -1 - eps:
            # print(f'[ datasets/mujoco ] Warning: sample out of range | ({x.min():.4f}, {x.max():.4f})')
            x = np.clip(x, -1, 1)

        ## [ -1, 1 ] --> [ 0, 1 ]
        x = (x + 1) / 2.

        return x * (self.maxs - self.mins) + self.mins

class SafeLimitsNormalizer(LimitsNormalizer):
    '''
        functions like LimitsNormalizer, but can handle data for which a dimension is constant
    '''

    def __init__(self, *args, eps=1, **kwargs):
        super().__init__(*args, **kwargs)
        for i in range(len(self.mins)):
            if self.mins[i] == self.maxs[i]:
                print(f'''
                    [ utils/normalization ] Constant data in dimension {i} | '''
                    f'''max = min = {self.maxs[i]}'''
                )
                self.mins -= eps
                self.maxs += eps

#-----------------------------------------------------------------------------#
#------------------------------- CDF normalizer ------------------------------#
#-----------------------------------------------------------------------------#

class CDFNormalizer(Normalizer):
    '''
        makes training data uniform (over each dimension) by transforming it with marginal CDFs
    '''

    def __init__(self, X):
        super().__init__(atleast_2d(X))
        self.dim = self.X.shape[1]
        self.cdfs = [
            CDFNormalizer1d(self.X[:, i])
            for i in range(self.dim)
        ]

    def __repr__(self):
        return f'[ CDFNormalizer ] dim: {self.mins.size}\n' + '    |    '.join(
            f'{i:3d}: {cdf}' for i, cdf in enumerate(self.cdfs)
        )

    def wrap(self, fn_name, x):
        shape = x.shape
        ## reshape to 2d
        x = x.reshape(-1, self.dim)
        out = np.zeros_like(x)
        for i, cdf in enumerate(self.cdfs):
            fn = getattr(cdf, fn_name)
            out[:, i] = fn(x[:, i])
        return out.reshape(shape)

    def normalize(self, x):
        return self.wrap('normalize', x)

    def unnormalize(self, x):
        return self.wrap('unnormalize', x)

class CDFNormalizer1d:
    '''
        CDF normalizer for a single dimension
    '''

    def __init__(self, X):
        assert X.ndim == 1
        self.X = X.astype(np.float32)
        quantiles, cumprob = empirical_cdf(self.X)
        self.fn = interpolate.interp1d(quantiles, cumprob)
        self.inv = interpolate.interp1d(cumprob, quantiles)

        self.xmin, self.xmax = quantiles.min(), quantiles.max()
        self.ymin, self.ymax = cumprob.min(), cumprob.max()

    def __repr__(self):
        return (
            f'[{np.round(self.xmin, 2):.4f}, {np.round(self.xmax, 2):.4f}'
        )

    def normalize(self, x):
        x = np.clip(x, self.xmin, self.xmax)
        ## [ 0, 1 ]
        y = self.fn(x)
        ## [ -1, 1 ]
        y = 2 * y - 1
        return y

    def unnormalize(self, x, eps=1e-4):
        '''
            X : [ -1, 1 ]
        '''
        ## [ -1, 1 ] --> [ 0, 1 ]
        x = (x + 1) / 2.

        if (x < self.ymin - eps).any() or (x > self.ymax + eps).any():
            print(
                f'''[ dataset/normalization ] Warning: out of range in unnormalize: '''
                f'''[{x.min()}, {x.max()}] | '''
                f'''x : [{self.xmin}, {self.xmax}] | '''
                f'''y: [{self.ymin}, {self.ymax}]'''
            )

        x = np.clip(x, self.ymin, self.ymax)

        y = self.inv(x)
        return y

def empirical_cdf(sample):
    ## https://stackoverflow.com/a/33346366

    # find the unique values and their corresponding counts
    quantiles, counts = np.unique(sample, return_counts=True)

    # take the cumulative sum of the counts and divide by the sample size to
    # get the cumulative probabilities between 0 and 1
    cumprob = np.cumsum(counts).astype(np.double) / sample.size

    return quantiles, cumprob

def atleast_2d(x):
    if x.ndim < 2:
        x = x[:,None]
    return x

