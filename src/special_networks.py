from rl_m.dataset import Dataset
from rl_m.typing import *
from rl_m.networks import *
import numpy as np
import torch.nn as nn
import torch
# from diffuser.ql_diffusion import Diffusion_QL

class LayerNormMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, activations=nn.GELU, activate_final=False, kernel_init=default_init):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.activations = activations
        self.activate_final = activate_final
        self.kernel_init = kernel_init
        self.layers = nn.ModuleList()
        for i, size in enumerate(hidden_dims[:-1]):
            self.layers.append(nn.Linear(input_dim if i == 0 else hidden_dims[i-1], size))
            self.layers.append(nn.LayerNorm(size))
            self.layers.append(activations())
        self.layers.append(nn.Linear(hidden_dims[-2], hidden_dims[-1]))
        if activate_final:
            self.layers.append(activations())

        self.model = nn.Sequential(*self.layers)
        self.apply(self.kernel_init)

    def forward(self, x):

        return self.model(x)



class LayerNormRepresentation(nn.Module):
    def __init__(self, input_dim, hidden_dims=(256, 256), activate_final=True, ensemble=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.activate_final = activate_final
        self.ensemble = ensemble
        kwargs = {
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'activate_final': self.activate_final,
        }
        self.module_cls = LayerNormMLP
        self.module = self.module_cls(**kwargs)
        
        if self.ensemble:
            self.module1 = self.module_cls(**kwargs)
            self.module2 = self.module_cls(**kwargs)

    def forward(self, observations):

        if self.ensemble:
            return self.module1(observations), self.module2(observations)
        return self.module(observations)


class Representation(nn.Module):

    def __init__(self,
                input_dim: int,
                hidden_dims: tuple = (256, 256),
                activate_final: bool = True,
                ensemble: bool = True) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.activate_final = activate_final
        self.ensemble = ensemble
        kwargs = {
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'activate_final': self.activate_final,
            'activations': nn.GELU,
        }
        module_cls = MLP
        self.module = module_cls(**kwargs)
        if self.ensemble:
            self.module1 = module_cls(**kwargs)
            self.module2 = module_cls(**kwargs)

    def forward(self, observations):
        if self.ensemble:
            return self.module1(observations), self.module2(observations)
        return self.module(observations)


class RelativeRepresentation(nn.Module):
    def __init__(self,
                input_dim: int,
                in_channel: int = 3,
                rep_dim: int = 256,
                hidden_dims: tuple = (256, 256),
                module: nn.Module = None,
                visual: bool = False,
                layer_norm: bool = False,
                rep_type: str = 'state',
                bottleneck: bool = True) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.rep_dim = rep_dim
        self.hidden_dims = (input_dim,) + hidden_dims
        self.module = module
        self.visual = visual
        self.layer_norm = layer_norm
        self.rep_type = rep_type
        self.bottleneck = bottleneck
        self.in_channels = in_channel

        if self.visual:
            self.v_module = self.module(in_channels=in_channel)
        if self.layer_norm:
            self.rep = LayerNormMLP(self.input_dim, self.hidden_dims, activate_final=not self.bottleneck, activations=nn.GELU)
        else:
            self.rep = MLP(self.input_dim, self.hidden_dims, activate_final=not self.bottleneck, activations=nn.GELU)

    def forward(self, targets, bases=None):
        if bases is None:
            inputs = targets
        else:
            if self.rep_type == 'state':
                inputs = targets
            elif self.rep_type == 'diff':
                inputs = torch.sub(targets, bases) + torch.ones_like(targets) * 1e-6
            elif self.rep_type == 'concat':
                inputs = torch.cat([targets, bases], dim=-1)
            else:
                raise NotImplementedError

        if self.visual:
            # Visual input
            # [B x H x W x C] -> [B x C x H x W]
            inputs = inputs.permute(0, 3, 1, 2)
            inputs = self.v_module(inputs)
        
        rep = self.rep(inputs)

        if self.bottleneck:
            rep = rep / torch.norm(rep, dim=-1, keepdim=True) * (np.sqrt(self.rep_dim).item())

        return rep


class MonolithicVF(nn.Module):

    def __init__(self,
                input_dim: int,
                hidden_dims = (256, 256),
                readout_size = (256,),
                use_layer_norm = True,
                rep_dim = None,
                obs_rep = 0,):
        super(MonolithicVF, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.readout_size = readout_size
        self.use_layer_norm = use_layer_norm
        self.rep_dim = rep_dim
        self.obs_rep = obs_rep
        repr_class = LayerNormRepresentation if self.use_layer_norm else Representation
        self.value_net = repr_class(input_dim, (*self.hidden_dims, 1), activate_final=False)

    def forward(self, observations, goals=None, info=False):
        phi = observations
        psi = goals
        v1, v2 = self.value_net(torch.cat([phi, psi], dim=-1))
        v1, v2 = v1.squeeze(-1), v2.squeeze(-1)
        if info:
            return {
                'v': (v1 + v2) / 2,
            }
        return v1, v2


def get_rep(
        encoder: nn.Module, targets: torch.Tensor, bases: torch.Tensor = None,
):
    if encoder is None:
        return targets
    else:
        if bases is None:
            return encoder(targets)
        else:
            return encoder(targets, bases)


class HierarchicalActorCritic(nn.Module):
    def __init__(self,
                encoders = {},
                networks = {},
                use_waypoints = 0,
                lr = 3e-4,
                device = "cpu") -> None:

        super().__init__()
        self.device = device
        self.encoders:Dict = encoders
        self.networks:Dict = networks
        self.use_waypoints = use_waypoints
        self.total_params = 0
        # to device
        for key, encoder in self.encoders.items():
            if encoder is not None:
                self.encoders[key] = encoder.to(self.device)
        for key, network in self.networks.items():
            self.networks[key] = network.to(self.device)

        self.optims = {}
        for key, encoder in self.encoders.items():
            if encoder is not None:
                self.optims[f"{key}_optim"] = torch.optim.Adam(encoder.parameters(), lr=lr)
                self.total_params += sum(p.numel() for p in encoder.parameters() if p.requires_grad)

        for key, network in self.networks.items():
            self.optims[f"{key}_optim"] = torch.optim.Adam(network.parameters(), lr=lr)
            self.total_params += sum(p.numel() for p in network.parameters() if p.requires_grad)

        print(f"Total number of parameters: {self.total_params/1e6:.2f}M")


    def value(self, observations, goals, encoded=False, **kwargs):

        state_reps = get_rep(self.encoders['value_state'], targets=observations)
        if not encoded:
            goal_reps = get_rep(self.encoders['value_goal'], targets=goals, bases=None)
        else:
            goal_reps = goals
        return self.networks['value'](state_reps, goal_reps, **kwargs)
    
    # @torch.no_grad()
    def target_value(self, observations, goals, **kwargs):
        state_reps = get_rep(self.encoders['value_state'], targets=observations)
        goal_reps = get_rep(self.encoders['value_goal'], targets=goals, bases=None)
        return self.networks['target_value'](state_reps, goal_reps, **kwargs)

    def actor(self, observations, goals, low_dim_goals=False, state_rep_grad=True, goal_rep_grad=True, eval=False,type=None, **kwargs):
        state_reps = get_rep(self.encoders['policy_state'], targets=observations)
        if not state_rep_grad:
            state_reps = state_reps.detach()

        if low_dim_goals:
            goal_reps = goals
        else:
            if self.use_waypoints:
                # Use the value_goal representation
                goal_reps = get_rep(self.encoders['value_goal'], targets=goals, bases=None)
            else:
                goal_reps = get_rep(self.encoders['policy_goal'], targets=goals, bases=observations)
            if not goal_rep_grad:
                goal_reps = goal_reps.detach()
        if type is None:
            return self.networks['actor'](torch.cat([state_reps, goal_reps], dim=-1), **kwargs)
        elif type == 'dql':
            cond = torch.cat([state_reps, goal_reps], dim=-1)
            if not eval:
                return self.networks['actor'].actor(cond, **kwargs)
            return self.networks['actor'].ema_model(cond, **kwargs)

    def high_actor(self, observations, goals, state_rep_grad=True, goal_rep_grad=True, eval=False, **kwargs):
        state_reps = get_rep(self.encoders['high_policy_state'], targets=observations)
        if not state_rep_grad:
            state_reps = state_reps.detach()

        goal_reps = get_rep(self.encoders['high_policy_goal'], targets=goals, bases=observations)
        if not goal_rep_grad:
            goal_reps = goal_reps.detach()
        # return self.networks['high_actor'](torch.cat([state_reps, goal_reps], dim=-1), **kwargs)

        return self.networks['high_actor'].actor(torch.cat([state_reps, goal_reps], dim=-1))

    def value_goal_encoder(self, targets, bases=None, **kwargs):
        return get_rep(self.encoders['value_goal'], targets=targets, bases=None)

    def value_state_encoder(self, targets, **kwargs):
        return get_rep(self.encoders['value_state'], targets=targets)
    
    def high_policy_goal_encoder(self, targets, bases, **kwargs):
        return get_rep(self.encoders['high_policy_goal'], targets=targets, bases=bases)

    def high_policy_state_encoder(self, targets, **kwargs):
        return get_rep(self.encoders['high_policy_state'], targets=targets)

    def policy_state_encoder(self, targets, **kwargs):
        return get_rep(self.encoders['policy_state'], targets=targets)

    # def value_goal_decoder(self, targets, bases, **kwargs):
    #     return get_rep(self.networks['goal_decoder'], targets=targets, bases=bases)

    def policy_goal_encoder(self, targets, bases, **kwargs):
        assert not self.use_waypoints
        return get_rep(self.encoders['policy_goal'], targets=targets, bases=bases)

    def forward(self, observations, goals):
        # Only for initialization
        rets = {
            'value': self.value(observations, goals),
            'target_value': self.target_value(observations, goals),
            'actor': self.actor(observations, goals),
            'high_actor': self.high_actor(observations, goals),
        }
        return rets
    
    def state_dict(self):
        state_dict = {}
        for key, encoder in self.encoders.items():
            if encoder is not None:
                state_dict[f"{key}_encoder"] = encoder.state_dict()
        for key, network in self.networks.items():
            state_dict[f"{key}_network"] = network.state_dict()
        return state_dict
    
    def load_state_dict(self, state_dict):
        for key, encoder in self.encoders.items():
            if encoder is not None:
                encoder.load_state_dict(state_dict[f"{key}_encoder"])
        for key, network in self.networks.items():
            network.load_state_dict(state_dict[f"{key}_network"])
    
    def to(self, device):
        for key, encoder in self.encoders.items():
            if encoder is not None:
                self.encoders[key] = encoder.to(device)
        for key, network in self.networks.items():
            self.networks[key] = network.to(device)
