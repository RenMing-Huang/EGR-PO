# from rl_m.dataset import Dataset
import torch
import dataclasses
import numpy as np
import ml_collections

@dataclasses.dataclass
class GCDataset:
    def __init__(self, dataset, p_randomgoal, p_trajgoal, p_currgoal, geom_sample, discount, terminal_key='dones_float', reward_scale=1.0, reward_shift=-1.0, terminal=True):
        self.dataset = dataset
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
    def __init__(self, dataset, p_randomgoal, p_trajgoal, p_currgoal, geom_sample, discount, way_steps=None, high_p_randomgoal=0., terminal_key='dones_float', reward_scale=1.0, reward_shift=0.0, terminal=False):
        super().__init__(dataset, p_randomgoal, p_trajgoal, p_currgoal, geom_sample, discount, terminal_key, reward_scale, reward_shift, terminal)
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

        batch['goals'] = self.dataset['achieved_goals'][goal_indx]

        final_state_indx = self.terminal_locs[np.searchsorted(self.terminal_locs, indx)]
        way_indx = np.minimum(indx + self.way_steps, final_state_indx)

        batch['low_goals'] = self.dataset['achieved_goals'][way_indx]

        distance = np.random.rand(batch_size)
        
        high_traj_goal_indx = np.round((np.minimum(indx + 1, final_state_indx) * distance + final_state_indx * (1 - distance))).astype(int)
        high_traj_target_indx = np.minimum(indx + self.way_steps, high_traj_goal_indx)

        high_random_goal_indx = np.random.randint(self.dataset.size, size=batch_size)
        high_random_target_indx = np.minimum(indx + self.way_steps, final_state_indx)

        pick_random = (np.random.rand(batch_size) < self.high_p_randomgoal)
        high_goal_idx = np.where(pick_random, high_random_goal_indx, high_traj_goal_indx)
        high_target_idx = np.where(pick_random, high_random_target_indx, high_traj_target_indx)

        batch['high_goals'] = self.dataset['achieved_goals'][high_goal_idx]
        batch['high_targets'] = self.dataset['achieved_goals'][high_target_idx]

        # if isinstance(batch['goals'], dict):
        #     from flax.core import freeze
        #     # Freeze the other observations
        #     batch['observations'] = freeze(batch['observations'])
        #     batch['next_observations'] = freeze(batch['next_observations'])

        return batch
