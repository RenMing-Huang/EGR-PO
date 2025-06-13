import copy
import numpy as np
from rl_m.typing import *
from rl_m.common import TrainState, target_update
from rl_m.networks import Policy, Critic, ensemblize, DiscretePolicy
import torch
import torch.nn.functional as F
import numpy as np
import ml_collections
from . import iql
from src.special_networks import Representation, HierarchicalActorCritic, RelativeRepresentation, MonolithicVF
from dql.agents.ql_diffusion import Diffusion_QL
from torch.distributions.normal import Normal
from torch.distributions import Independent
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
log_std_min=-20
log_std_max=2

def optimize(loss:torch.Tensor, agent:"JointTrainAgent", optims):
    optimizers = []
    for optim in optims:
        # check if the optimizer is in the network
        if optim in agent.network.optims:
            optimizers.append(agent.network.optims[optim])
    for optim in optimizers:
        optim.zero_grad()

    loss.backward()

    for optim in optimizers:
        optim.step()



def expectile_loss(adv, diff, expectile=0.7):
    weight = torch.where(adv >= 0, expectile, (1 - expectile))
    return weight * (diff**2)


def compute_actor_loss(agent:"JointTrainAgent", batch):
    # actually, we don't use this
    if agent.config['use_waypoints']:  # Use waypoint states as goals (for hierarchical policies)
        cur_goals = batch['low_goals']
    else:  # Use randomized last observations as goals (for flat policies)
        cur_goals = batch['high_goals']

    with torch.no_grad():
        v1, v2 = agent.network.value(batch['achieved_goals'], cur_goals) # s, s_sub
        nv1, nv2 = agent.network.value(batch['next_achieved_goals'], cur_goals) # s', s_sub

    v = (v1 + v2) / 2
    nv = (nv1 + nv2) / 2
    adv = nv - v
    exp_a = torch.exp(adv * agent.config['temperature'])
    exp_a = torch.clamp(exp_a, max=100.0)

    if agent.config['use_waypoints']:
        goal_rep_grad = agent.config['policy_train_rep']
    else:
        goal_rep_grad = True

    dist = agent.network.actor(batch['observations'], cur_goals, state_rep_grad=True, goal_rep_grad=goal_rep_grad)
    log_probs = dist.log_prob(batch['actions'])
    actor_loss = -(exp_a * log_probs).mean()

    # optims = ['actor_optim', 'policy_state_optim' , 'policy_goal_optim']
    optims = ['actor_optim', 'policy_state_optim']
    optimize(actor_loss, agent, optims)

    actor_loss = actor_loss.detach().cpu().numpy()
    adv = adv.detach().cpu().numpy()
    mse = torch.mean((dist.mean - batch['actions'])**2).detach().cpu().numpy()
    std = dist.stddev.mean().detach().cpu().numpy()  # Fix: Access the 'stddev' attribute instead of 'scale'
    return actor_loss, {
        'actor_loss': actor_loss,
        'adv': adv.mean(),
        'bc_log_probs': log_probs.mean(),
        'adv_median': np.median(adv),
        'mse': mse,
        'std': std,  # Fix: Use the 'log_std' variable instead of 'dist.scale.mean().detach().cpu().numpy()'
    }


def compute_high_actor_loss(agent:"JointTrainAgent", batch, pred_goal=None):
    high_policy:Diffusion_QL = agent.network.networks['high_actor']
    cur_goals = batch['high_goals']

    with torch.no_grad():
        v1, v2 = agent.network.value(batch['observations'], cur_goals) # s, g
        nv1, nv2 = agent.network.value(batch['high_targets'], cur_goals) # s_sub, g

    v = (v1 + v2) / 2
    nv = (nv1 + nv2) / 2

    adv = nv - v
    exp_a = torch.exp(adv * agent.config['high_temperature'])
    exp_a = torch.clamp(exp_a, max=100.0).unsqueeze(-1)

    
    if agent.config['use_rep']:
        with torch.no_grad():
            target = agent.network.value_goal_encoder(targets=batch['high_targets'], bases=batch['observations'])
    else:
        if pred_goal is  None:
            target = batch['high_targets']
        elif pred_goal == 'diff':
            target = batch['high_targets'] - batch['observations']
        
    
    bc_loss = high_policy.actor.loss(target, torch.cat([batch['observations'], cur_goals], dim=-1))

    generated_target = high_policy.actor(torch.cat([batch['observations'],  cur_goals], dim=-1))

    q_loss = exp_a * F.mse_loss(generated_target, target, reduction='none')
   
    loss = bc_loss + q_loss.mean()

    high_policy.actor_optimizer.zero_grad()
    loss.backward()
    high_policy.actor_optimizer.step()

    return loss, {
        'behavior_loss': bc_loss.detach().cpu().numpy(),
        'q_loss': q_loss.mean().detach().cpu().numpy(),
    }


def compute_value_loss(agent:"JointTrainAgent", batch):
    """
    This can learn a value function for any goal, but we only use it for the current goal.
    """
    # masks are 0 if terminal, 1 otherwise
    batch['masks'] = 1.0 - batch['rewards']
    # rewards are 0 if terminal, -1 otherwise
    batch['rewards'] = batch['rewards'] - 1.0
    with torch.no_grad():
        (next_v1, next_v2) = agent.network.target_value(batch['next_observations'], batch['goals'])
        (v1_t, v2_t) = agent.network.target_value(batch['observations'], batch['goals'])

    next_v = torch.minimum(next_v1, next_v2)
    q = batch['rewards'] + agent.config['discount'] * batch['masks'] * next_v

    v_t = (v1_t + v2_t) / 2
    adv = q - v_t

    q1 = batch['rewards'] + agent.config['discount'] * batch['masks'] * next_v1
    q2 = batch['rewards'] + agent.config['discount'] * batch['masks'] * next_v2
    (v1, v2) = agent.network.value(batch['observations'], batch['goals'])

    value_loss1 = expectile_loss(adv, q1 - v1, agent.config['pretrain_expectile']).mean()
    value_loss2 = expectile_loss(adv, q2 - v2, agent.config['pretrain_expectile']).mean()
    value_loss = value_loss1 + value_loss2
    
    optims = ['value_optim', 'value_state_optim', 'value_goal_optim']
    optimize(value_loss, agent, optims)
    value_loss = value_loss.detach().cpu().numpy()
    advantage = adv.detach().cpu().numpy()

    return value_loss, {
        'value_loss': value_loss,
        'v max': v1.max(),
        'v min': v1.min(),
        'v mean': v1.mean(),
        'abs adv mean': np.abs(advantage).mean(),
        'adv mean': advantage.mean(),
        'adv max': advantage.max(),
        'adv min': advantage.min(),
        'accept prob': (advantage >= 0).mean(),
    }




class JointTrainAgent(iql.IQLAgent):
    def __init__(self, rng, network, critic, value, target_value, actor, config):
        super().__init__(rng, critic, value, target_value, actor, config)
        self.network:HierarchicalActorCritic = network

    def pretrain_update(agent, pretrain_batch, seed=None, value_update=True, actor_update=True, high_actor_update=True, pred_goal=None):
        # to tensor
        for k, v in pretrain_batch.items():
            pretrain_batch[k] = torch.tensor(v).float().to(device)
        def loss_fn():
            info = {}
            # Value
            if value_update:
                value_loss, value_info = compute_value_loss(agent, pretrain_batch)
                for k, v in value_info.items():
                    info[f'value/{k}'] = v
            else:
                value_loss = 0.

            # Actor
            if actor_update:
                actor_loss, actor_info = compute_actor_loss(agent, pretrain_batch)
                for k, v in actor_info.items():
                    info[f'actor/{k}'] = v
            else:
                actor_loss = 0.

            # High Actor
            if high_actor_update and agent.config['use_waypoints']:
                high_actor_loss, high_actor_info = compute_high_actor_loss(agent, pretrain_batch, pred_goal)
                for k, v in high_actor_info.items():
                    info[f'high_actor/{k}'] = v
            else:
                high_actor_loss = 0.



            loss = {"value_loss": value_loss, "actor_loss": actor_loss, "high_actor_loss": high_actor_loss}

            return loss, info

        loss, info = loss_fn()

        # update target
        if value_update:
            target_update(agent.network.networks["value"], agent.network.networks["target_value"], agent.config['target_update_rate'])

        return agent, loss, info

    @torch.no_grad()
    def sample_actions(agent,
                       observations: torch.Tensor,
                       goals: torch.Tensor,
                       *,
                       low_dim_goals: bool = False,
                       seed:int=0 ,
                       temperature: float = 1.0,
                       discrete: int = 0,
                       num_samples: int = None) -> np.ndarray:
        dist = agent.network.actor(observations, goals, low_dim_goals=low_dim_goals, temperature=temperature)
        if num_samples is None:
            actions = dist.mean
        else:
            actions = dist.sample((num_samples,))
        if not discrete:
            actions = torch.clamp(actions, -1, 1)
        return actions.squeeze().cpu().numpy()
        # actions = agent.network.actor(observations, goals, low_dim_goals=low_dim_goals, temperature=temperature)
        # return actions.squeeze().cpu().numpy()
    
    @torch.no_grad()
    def sample_high_actions(agent,
                            observations: torch.Tensor,
                            goals: torch.Tensor,
                            *,
                            seed:int=0,
                            temperature: float = 1.0,
                            num_samples: int = None,
                            return_mean = False) -> torch.Tensor:
        
        if isinstance(observations, np.ndarray):
            observations = torch.tensor(observations).unsqueeze(0).float().to(device)
        if isinstance(goals, np.ndarray):
            goals = torch.tensor(goals).unsqueeze(0).float().to(device)
        batch_size = observations.shape[0]
        if num_samples:
            observations = torch.repeat_interleave(observations, num_samples, dim=0)
            goals = torch.repeat_interleave(goals, num_samples, dim=0)
        actions = agent.network.high_actor(observations, goals, temperature=temperature, eval=True)
        reshaped = actions.view(batch_size, -1, actions.shape[-1])
        if return_mean:
            return reshaped.mean(dim=1)
        return actions.squeeze(1)

        

    @torch.no_grad()
    def get_policy_rep(agent,
                       *,
                       targets: torch.Tensor,
                       bases: torch.Tensor = None,
                       ) -> torch.Tensor:
        return agent.network.policy_goal_encoder(targets=targets, bases=bases)


import torch
import torch.nn as nn
import torch.optim as optim

def create_learner(
        seed: int,
        observations: torch.Tensor,
        actions: torch.Tensor,
        lr: float = 3e-4,
        actor_hidden_dims: Sequence[int] = (256, 256),
        value_hidden_dims: Sequence[int] = (256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        temperature: float = 1,
        high_temperature: float = 1,
        pretrain_expectile: float = 0.7,
        way_steps: int = 0,
        rep_dim: int = 10,
        use_rep: int = 0,
        policy_train_rep: float = 0,
        visual: int = 0,
        encoder: str = 'impala',
        discrete: int = 0,
        use_layer_norm: int = 0,
        rep_type: str = 'state',
        use_waypoints: int = 0,
        ha_eta: float = 0.5,
        goal_dim: int = None,
        pred_goal:str = None,
        **kwargs):

        print('Extra kwargs:', kwargs)

        if goal_dim is  None:
            goal_dim = observations.shape[-1]

        obs_dim = observations.shape[-1]
        print('Observation dim:', obs_dim)

        rng = np.random.default_rng(seed)
        actor_key, high_actor_key, critic_key, value_key = rng.integers(0, 2**32, size=(4,))

        

        value_state_encoder = None
        value_goal_encoder = None
        policy_state_encoder = None
        policy_goal_encoder = None
        high_policy_state_encoder = None
        high_policy_goal_encoder = None
        if visual:
            assert use_rep
            from rl_m.vision import encoders
            obs_dim = 16 * 8 * 8

            visual_encoder = encoders[encoder]
            def make_encoder(bottleneck, in_chan=3):
                if bottleneck:
                    return RelativeRepresentation(input_dim = obs_dim, in_channel=in_chan, rep_dim=rep_dim, hidden_dims=(*value_hidden_dims[:-1],rep_dim), visual=True, module=visual_encoder, layer_norm=use_layer_norm, rep_type=rep_type, bottleneck=True)
                else:
                    return RelativeRepresentation(input_dim = obs_dim, in_channel=in_chan, rep_dim=value_hidden_dims[-1], hidden_dims=(*value_hidden_dims,), visual=True, module=visual_encoder, layer_norm=use_layer_norm, rep_type=rep_type, bottleneck=False)

            value_state_encoder = make_encoder(bottleneck=False)
            value_goal_encoder = make_encoder(bottleneck=use_waypoints, in_chan=6)
            policy_state_encoder = make_encoder(bottleneck=False)
            policy_goal_encoder = make_encoder(bottleneck=False, in_chan=6)
            high_policy_state_encoder = make_encoder(bottleneck=False)
            high_policy_goal_encoder = make_encoder(bottleneck=False, in_chan=6)
        else:
            def make_encoder(bottleneck):
                if bottleneck:
                    return RelativeRepresentation(input_dim = obs_dim, rep_dim=rep_dim, hidden_dims=(*value_hidden_dims[:-1], rep_dim), layer_norm=use_layer_norm, rep_type=rep_type, bottleneck=True)
                else:
                    return RelativeRepresentation(input_dim = obs_dim, rep_dim=value_hidden_dims[-1], hidden_dims=(*value_hidden_dims, ), layer_norm=use_layer_norm, rep_type=rep_type, bottleneck=False)

            if use_rep:
                value_goal_encoder = make_encoder(bottleneck=True)

        if visual:
            input_dim = value_hidden_dims[-1] if value_state_encoder else obs_dim
            input_dim += rep_dim if value_goal_encoder else obs_dim
        else:
            input_dim = rep_dim if value_state_encoder else obs_dim # whether obs encoder
            input_dim += rep_dim if value_goal_encoder else goal_dim # whether goal encoder


        value_def = MonolithicVF(input_dim = input_dim, hidden_dims=value_hidden_dims, use_layer_norm=use_layer_norm, rep_dim=rep_dim)
        target_value_def = copy.deepcopy(value_def)
        # freeze target value
        for p in target_value_def.parameters():
            p.requires_grad = False

        if visual:
            input_dim = value_hidden_dims[-1] if policy_state_encoder else obs_dim # whether obs encoder
            input_dim += rep_dim if use_waypoints else value_hidden_dims[-1] # whether goal encoder
        else:
            input_dim = rep_dim if policy_state_encoder else obs_dim # whether obs encoder
            input_dim += rep_dim if policy_goal_encoder or (use_waypoints and value_goal_encoder) else obs_dim # whether goal encoder

        
        if discrete:
            action_dim = actions[0] + 1
            actor_def = DiscretePolicy(input_dim, actor_hidden_dims, action_dim=action_dim)
        else:
            action_dim = actions.shape[-1]
            actor_def = Policy(input_dim, actor_hidden_dims, action_dim=action_dim, log_std_min=-5.0, state_dependent_std=False, tanh_squash_distribution=False)

        if visual:
            input_dim = value_hidden_dims[-1] if high_policy_state_encoder else obs_dim # whether obs encoder
            input_dim += value_hidden_dims[-1] if high_policy_goal_encoder or use_waypoints else obs_dim # whether goal encoder
        else:
            input_dim = rep_dim if high_policy_state_encoder else obs_dim # whether obs encoder
            input_dim += rep_dim if high_policy_goal_encoder else goal_dim # whether goal encoder

        high_action_dim = obs_dim if not use_rep else rep_dim
        high_actor_def = Diffusion_QL(
            state_dim=input_dim,
            goal_dim=high_action_dim,
            discount=discount,
            max_action=None,
            beta_schedule="vp",
            n_timesteps=5,
            device=device,
            lr=lr,
            tau=0.005,
            eta=ha_eta,
        )


        network_def = HierarchicalActorCritic(
            encoders={
                'value_state': value_state_encoder,
                'value_goal': value_goal_encoder,
                'policy_state': policy_state_encoder,
                'policy_goal': policy_goal_encoder,
                'high_policy_state': high_policy_state_encoder,
                'high_policy_goal': high_policy_goal_encoder,
            },
            networks={
                'value': value_def,
                'target_value': target_value_def,
                'actor': actor_def,
                'high_actor': high_actor_def,
            },
            use_waypoints=use_waypoints,
            lr=lr,
            device = device,
        )
        # init params
        
        network = network_def

        config = {
            'discount': discount,
            'temperature': temperature,
            'high_temperature': high_temperature,
            'target_update_rate': tau,
            'pretrain_expectile': pretrain_expectile,
            'way_steps': way_steps,
            'rep_dim': rep_dim,
            'policy_train_rep': policy_train_rep,
            'use_rep': use_rep,
            'use_waypoints': use_waypoints,
            'lr': lr,
        }

        return JointTrainAgent(rng, network=network, critic=None, value=None, target_value=None, actor=None, config=config)


def get_default_config():
    config = ml_collections.ConfigDict({
        'lr': 3e-4,
        'actor_hidden_dims': (256, 256),
        'value_hidden_dims': (256, 256),
        'discount': 0.99,
        'temperature': 1.0,
        'tau': 0.005,
        'pretrain_expectile': 0.7,
    })

    return config