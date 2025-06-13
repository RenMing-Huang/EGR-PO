from collections import namedtuple
import numpy as np
import torch
import d4rl

from .preprocessing import get_preprocess_fn
from .d4rl import load_environment, sequence_dataset
from src.d4rl_utils import get_dataset
from .normalization import DatasetNormalizer
from .buffer import ReplayBuffer

RewardBatch = namedtuple('Batch', 'observations next_observations actions rtg')
Batch = namedtuple('Batch', 'observations next_observations actions')
ValueBatch = namedtuple('ValueBatch', 'trajectories conditions values')

# from rl_m.dataset import Dataset
import torch
import dataclasses
import numpy as np
import ml_collections

@dataclasses.dataclass
class GCDataset:
    def __init__(self, env_name, p_randomgoal, p_trajgoal, p_currgoal, geom_sample, discount, terminal_key='dones_float', reward_scale=1.0, reward_shift=-1.0, terminal=True):
        self.dataset = get_dataset(load_environment(env_name), env_name)
        self.p_randomgoal = p_randomgoal
        self.p_trajgoal = p_trajgoal
        self.p_currgoal = p_currgoal
        self.geom_sample = geom_sample
        self.discount = discount
        self.terminal_key = terminal_key
        self.reward_scale = reward_scale
        self.reward_shift = reward_shift
        self.terminal = terminal
        self.terminal_locs, = np.nonzero(self.dataset[self.terminal_key] > 0)
        assert torch.isclose(torch.tensor(self.p_randomgoal + self.p_trajgoal + self.p_currgoal), torch.tensor(1.0))

    @staticmethod
    def get_default_config():
        return {
            'p_randomgoal': 0.3,
            'p_trajgoal': 0.5,
            'p_currgoal': 0.2,
            'geom_sample': 0,
            'reward_scale': 1.0,
            'reward_shift': -1.0,
            'terminal': True,
        }

    def sample_goals(self, indx, p_randomgoal=None, p_trajgoal=None, p_currgoal=None):
        if p_randomgoal is None:
            p_randomgoal = self.p_randomgoal
        if p_trajgoal is None:
            p_trajgoal = self.p_trajgoal
        if p_currgoal is None:
            p_currgoal = self.p_currgoal

        batch_size = len(indx)
        # self.dataset.size = len(self.dataset['observations'])
        # Random goals
        goal_indx = np.random.randint(self.dataset.size, size=batch_size)
        
        # Goals from the same trajectory
        final_state_indx = self.terminal_locs[np.searchsorted(self.terminal_locs, indx)]

        distance = np.random.rand(batch_size)
        if self.geom_sample:
            us = np.random.rand(batch_size)
            middle_goal_indx = np.minimum(indx + np.ceil(np.log(1 - us) / np.log(self.discount)).astype(int), final_state_indx)
        else:
            middle_goal_indx = np.round((np.minimum(indx + 1, final_state_indx) * distance + final_state_indx * (1 - distance))).astype(int)

        goal_indx = np.where(np.random.rand(batch_size) < p_trajgoal / (1.0 - p_currgoal), middle_goal_indx, goal_indx)
        
        # Goals at the current state
        goal_indx = np.where(np.random.rand(batch_size) < p_currgoal, indx, goal_indx)
        return goal_indx

    def sample(self, batch_size: int, indx=None):
        # self.dataset.size = len(self.dataset['observations'])
        if indx is None:
            indx = torch.randint(self.dataset.size-1, size=(batch_size,))

        batch = self.dataset.sample(batch_size, indx)
        goal_indx = self.sample_goals(indx)

        success = (indx == goal_indx)
        batch['rewards'] = success.to(torch.float) * self.reward_scale + self.reward_shift
        if self.terminal:
            batch['masks'] = (1.0 - success.to(torch.float))
        else:
            batch['masks'] = torch.ones(batch_size)
        # batch['goals'] = jax.tree_map(lambda arr: arr[goal_indx], self.dataset['observations'])
        batch['goals'] = {}
        for key, value in self.dataset['observations'].items():
            batch['goals'][key] = value[goal_indx]

        return batch

class GCSDataset(GCDataset):
    def __init__(self, env_name, p_randomgoal=0.2, p_trajgoal=0.5, p_currgoal=0.3, geom_sample=1, discount=0.99, way_steps=25, high_p_randomgoal=0., terminal_key='dones_float', reward_scale=1.0, reward_shift=0.0, terminal=False):
        super().__init__(env_name, p_randomgoal, p_trajgoal, p_currgoal, geom_sample, discount, terminal_key, reward_scale, reward_shift, terminal)
        self.way_steps = way_steps
        self.high_p_randomgoal = high_p_randomgoal

    @staticmethod
    def get_default_config():
        return ml_collections.ConfigDict({
            'p_randomgoal': 0.3,
            'p_trajgoal': 0.5,
            'p_currgoal': 0.2,
            'geom_sample': 0,
            'reward_scale': 1.0,
            'reward_shift': 0.0,
            'terminal': False,
        })

    def sample(self, batch_size: int, indx=None):
        # self.dataset.size = len(self.dataset['observations'])
        if indx is None:
            indx = np.random.randint(self.dataset.size-1, size=batch_size)

        batch = self.dataset.sample(batch_size, indx)
        goal_indx = self.sample_goals(indx)

        success = (indx == goal_indx)
        batch['rewards'] = success.astype(float) * self.reward_scale + self.reward_shift
        if self.terminal:
            batch['masks'] = (1.0 - success.astype(float))
        else:
            batch['masks'] = np.ones(batch_size)

        batch['goals'] = self.dataset['observations'][goal_indx]

        final_state_indx = self.terminal_locs[np.searchsorted(self.terminal_locs, indx)]
        way_indx = np.minimum(indx + self.way_steps, final_state_indx)

        batch['low_goals'] = self.dataset['observations'][way_indx]

        distance = np.random.rand(batch_size)
        
        high_traj_goal_indx = np.round((np.minimum(indx + 1, final_state_indx) * distance + final_state_indx * (1 - distance))).astype(int)
        high_traj_target_indx = np.minimum(indx + self.way_steps, high_traj_goal_indx)

        high_random_goal_indx = np.random.randint(self.dataset.size, size=batch_size)
        high_random_target_indx = np.minimum(indx + self.way_steps, final_state_indx)

        pick_random = (np.random.rand(batch_size) < self.high_p_randomgoal)
        high_goal_idx = np.where(pick_random, high_random_goal_indx, high_traj_goal_indx)
        high_target_idx = np.where(pick_random, high_random_target_indx, high_traj_target_indx)

        batch['high_goals'] = self.dataset['observations'][high_goal_idx]
        batch['high_targets'] = self.dataset['observations'][high_target_idx]

        # if isinstance(batch['goals'], dict):
        #     from flax.core import freeze
        #     # Freeze the other observations
        #     batch['observations'] = freeze(batch['observations'])
        #     batch['next_observations'] = freeze(batch['next_observations'])

        return batch

# def sequence_dataset(env, data):
#     """
#     Returns an iterator through trajectories.
#     Args:
#         env: An OfflineEnv object.
#         dataset: An optional dataset to pass in for processing. If None,
#             the dataset will default to env.get_dataset()
#         **kwargs: Arguments to pass to env.get_dataset().
#     Returns:
#         An iterator through dictionaries with keys:
#             observations
#             actions
#             rewards
#             terminals
#     """
#     dataset = data

#     N = dataset['rewards'].shape[0]
#     data_ = collections.defaultdict(list)

#     # The newer version of the dataset adds an explicit
#     # timeouts field. Keep old method for backwards compatability.
#     use_timeouts = 'timeouts' in dataset
#     if not use_timeouts:
#         dataset['timeouts'] = np.zeros_like(dataset['terminals'])
#     episode_step = 0
#     for i in range(N):
#         done_bool = bool(dataset['terminals'][i])
#         if use_timeouts:
#             final_timestep = dataset['timeouts'][i]
#         else:
#             final_timestep = (episode_step == env._max_episode_steps - 1)
#             dataset['timeouts'][i] = final_timestep
    
#         for k in dataset:
#             if 'metadata' in k: continue
#             data_[k].append(dataset[k][i])

#         if done_bool or final_timestep:
#             episode_step = 0
#             episode_data = {}
#             for k in data_:
#                 episode_data[k] = np.array(data_[k])
#             yield episode_data
#             data_ = collections.defaultdict(list)

#         episode_step += 1

class SequenceDatasetV2(torch.utils.data.Dataset):
    def __init__(self, 
                env_name,
                normalizer='GaussianNormalizer',
                use_normalizer=True,
                preprocess_fns=[], 
                horizon=16, 
                max_n_episodes=20000, 
                termination_penalty=0, 
                use_padding=False, 
                discount=1, 
                returns_scale=1000, 
                include_returns=True,
                value_model=None,
                device = 'cpu'):
        self.device = device
        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env_name)
        self.env = env = load_environment(env_name)
        # get the max length of env
        self.max_path_length = 1000
        self.horizon = horizon
        self.returns_scale = returns_scale
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length, dtype=np.float32)[:, None]
        self.use_padding = use_padding
        self.include_returns = include_returns
        itr = sequence_dataset(env, self.preprocess_fn)

        fields = ReplayBuffer(max_n_episodes, self.max_path_length, termination_penalty, self.discounts)
        for i, episode in enumerate(itr):
            fields.add_path(episode)
        fields.finalize()

        self.normalizer = DatasetNormalizer(fields, normalizer, path_lengths=fields['path_lengths'])
        self.indices = self.make_indices(fields.path_lengths, horizon)

        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1]
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths
        if use_normalizer:
            self.normalize(keys=["observations", "next_observations"])

        self.value_model = value_model
        print(fields)

    def get_rewards(self, obs_goal_pair):
        
        obs = obs_goal_pair[:-1].reshape(-1, self.observation_dim)
        goal = obs_goal_pair[1:].reshape(-1, self.observation_dim)
        # to tensor
        obs = torch.tensor(obs, dtype=torch.float32)
        goal = torch.tensor(goal, dtype=torch.float32)
        r1, r2 = self.value_model(obs, goal)
        r = (r1 + r2) / 2
        return r.detach().numpy()

    def re_make_indices(self):
        self.indices = self.make_indices(self.fields.path_lengths, self.horizon)

    def normalize(self, keys):
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)

    def make_indices(self, path_lengths, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            for j in range(path_length - horizon):
                indices.append((i, j, j + horizon))
        indices = np.array(indices)
        return indices

    def get_conditions(self, observations):
        '''
            condition on current observation for planning
        '''
        return {0: observations[0]}
    
    def sample(self, batch_size):
        '''
            sample from dataset
        '''
        # indices = np.random.choice(len(self.path_lengths), batch_size, replace=True)
        # path_lengths = self.path_lengths[indices]
        # # 对每个path,从0-pathlength中随机选取horizon个不连续的中间状态，不要求连续
        # indexs = []
        # for _, path_length in enumerate(path_lengths):
        #     random_index = np.random.choice(path_length, self.horizon, replace=False)
        #     indexs.append(sorted(random_index))

        # indexs = np.array(indexs)
        # index = np.random.choice(len(self.path_lengths), batch_size, replace=False)
        # return self.fields.observations[index], self.fields.discounted_rewards[index]
        pass



    def __len__(self):
        return len(self.path_lengths)

    def __getitem__(self, idx, eps=1e-4):
        # path_ind, start, end = self.indices[idx]
        length = self.path_lengths[idx]
        assert length > self.horizon, f'path length {length} <= horizon {self.horizon}'
        
        # sample goal from trajectory
        goal_index = np.random.randint(self.horizon, length - self.horizon)
        goals = self.fields.normed_observations[idx, goal_index].reshape(-1, self.observation_dim)
        # sample sub_goal towards the goal
        sub_goal_index = np.random.randint(0, goal_index, self.horizon)
        sub_goal_index = np.sort(sub_goal_index)
        observations = self.fields.normed_observations[idx, sub_goal_index]
        obs_goal_pair = np.concatenate([observations, goals], axis=0)
        discounted_rewards = self.get_rewards(obs_goal_pair) / self.returns_scale

        transition = np.concatenate([discounted_rewards[:, np.newaxis], observations], axis=1)
        cond = self.get_conditions(observations)
        conditions = goals
        
        # # uniform sampling
        # index = np.arange(random_start, random_start + offset * self.horizon, offset)
        # assert len(index) == self.horizon , f'index length {len(index)} != horizon {self.horizon}'
        # observations = self.fields.normed_observations[idx, index]
        # sum_rewards = self.fields.sum_rewards[idx, index]
        # rewards = (sum_rewards[1:] - sum_rewards[:-1]) / self.returns_scale
        # discounted_reward = self.fields.discounted_rewards[idx] / self.returns_scale
        # rewards = np.concatenate([discounted_reward[np.newaxis], rewards])

        # transition = np.concatenate([rewards[:, np.newaxis], observations], axis=1)
        # cond = self.get_conditions(observations)
        # next_observations = self.fields.normed_next_observations[path_ind, start:end]


        # if self.include_returns:
        #     # bug fixed
        #     rewards = self.fields.rewards[path_ind, start:length]
        #     discounts = self.discounts[:len(rewards)]
        #     returns = np.cumsum(discounts * rewards[::-1])[::-1][:self.horizon] / self.returns_scale
        #     return observations, next_observations, actions, returns

        return transition, cond, conditions


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, itr, horizon, state_mean, state_std, scale,device, penalty=None) -> None:
        super().__init__()

        self.device = device
        self.horizon = horizon
        self.termination_penalty = penalty

        self.observations = []
        self.actions = []
        self.next_observations = []
        self.rewards = []
        self.rtg = []
        self.terminals = []
        self.reward_to_go = []
        discount = 1


        for i, episode in enumerate(itr):
            # reward to go
            rtg = []

            episode_return = 0.0

            # penalty last step
            if self.termination_penalty is not None and episode['terminals'][-1] and not episode['timeouts'][-1]:
                episode['rewards'][-1] += self.termination_penalty


            episode["rewards"] = episode["rewards"] / scale
            for i in range(len(episode["rewards"])-1,-1,-1):
                episode_return = episode["rewards"][i] + episode_return * discount
                rtg.append(episode_return)

            rtg.reverse()

            # self.max_reward = max(self.max_reward, rtg[0])

            for j in range(0, len(episode["observations"]) - horizon + 1, 1):
                self.observations.append(episode["observations"][j:j+horizon])
                self.actions.append(episode["actions"][j:j+horizon])
                self.next_observations.append(episode["next_observations"][j:j+horizon])
                self.rewards.append(rtg[j:j+horizon])
                self.terminals.append(episode["terminals"][j:j+horizon])
                self.reward_to_go.append(rtg[0])



        # normalize
        self.observations = (np.array(self.observations) - state_mean) / (state_std + 1e-8)
        self.next_observations = (np.array(self.next_observations) - state_mean) / (state_std + 1e-8)

        # to tensor
        self.observations = torch.from_numpy(self.observations).float()
        self.actions = torch.tensor(np.array(self.actions), dtype=torch.float32)
        self.next_observations = torch.tensor(np.array(self.next_observations), dtype=torch.float32)
        self.rewards = torch.tensor(np.array(self.rewards), dtype=torch.float32)
        self.terminals = torch.tensor(np.array(self.terminals), dtype=torch.float32)
        self.reward_to_go = torch.tensor(np.array(self.reward_to_go), dtype=torch.float32)

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, index):
        return (
            self.observations[index],
            self.actions[index],
            self.next_observations[index],
            self.rewards[index],
            self.terminals[index],
            self.reward_to_go[index],
            
        )

def iql_normalize(reward, not_done):
	trajs_rt = []
	episode_return = 0.0
	for i in range(len(reward)):
		episode_return += reward[i]
		if not not_done[i]:
			trajs_rt.append(episode_return)
			episode_return = 0.0
	rt_max, rt_min = torch.max(torch.tensor(trajs_rt)), torch.min(torch.tensor(trajs_rt))
	reward /= (rt_max - rt_min)
	reward *= 1000.
	return reward

class NormalDataset(torch.utils.data.Dataset):
        def __init__(self, data, device, reward_tune='normalize'):
            state = torch.from_numpy(data['observations']).float()
            action = torch.from_numpy(data['actions']).float()
            next_state = torch.from_numpy(data['next_observations']).float()
            #normalize
            self.state_mean, self.state_std = state.mean(dim=0), state.std(dim=0)
            # self.action_mean, self.action_std = action.mean(dim=0), action.std(dim=0)
            self.state = (state - state.mean(dim=0)) / state.std(dim=0)
            self.next_state = (next_state - state.mean(dim=0)) / state.std(dim=0)
            self.action = action

            
            reward = torch.from_numpy(data['rewards']).view(-1, 1).float()
            self.done = torch.from_numpy(data['terminals']).view(-1, 1).float()
            self.size = self.state.shape[0]
            self.state_dim = self.state.shape[1]
            self.action_dim = self.action.shape[1]

            self.device = device

            if reward_tune == 'normalize':
                reward = (reward - reward.mean()) / reward.std()
            elif reward_tune == 'iql_antmaze':
                reward = reward - 1.0
            elif reward_tune == 'iql_locomotion':
                reward = iql_normalize(reward, self.not_done)
            elif reward_tune == 'cql_antmaze':
                reward = (reward - 0.5) * 4.0
            elif reward_tune == 'antmaze':
                reward = (reward - 0.25) * 2.0
            self.reward = reward
        def __len__(self):
            return self.size
        
        def __getitem__(self, ind):

            return (
                self.state[ind].to(self.device),
                self.action[ind].to(self.device),
                self.next_state[ind].to(self.device),
                self.reward[ind].to(self.device),
                self.done[ind].to(self.device)
            )

        def get_states(self):
            return self.state_mean, self.state_std