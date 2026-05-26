from rl_m.dataset import Dataset
from rl_m.typing import *
from rl_m.networks import *
import numpy as np
import torch.nn as nn
import torch


class LayerNormMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, activations=nn.GELU, activate_final=False, kernel_init=default_init):
        super().__init__()
        self.layers = nn.ModuleList()
        for i, size in enumerate(hidden_dims[:-1]):
            self.layers.append(nn.Linear(input_dim if i == 0 else hidden_dims[i - 1], size))
            self.layers.append(nn.LayerNorm(size))
            self.layers.append(activations())
        self.layers.append(nn.Linear(hidden_dims[-2], hidden_dims[-1]))
        if activate_final:
            self.layers.append(activations())
        self.model = nn.Sequential(*self.layers)
        self.apply(kernel_init)

    def forward(self, x):
        return self.model(x)


class LayerNormRepresentation(nn.Module):
    def __init__(self, input_dim, hidden_dims=(256, 256), activate_final=True, ensemble=True):
        super().__init__()
        self.ensemble = ensemble
        kwargs = {'input_dim': input_dim, 'hidden_dims': hidden_dims, 'activate_final': activate_final}
        self.module = LayerNormMLP(**kwargs)
        if ensemble:
            self.module1 = LayerNormMLP(**kwargs)
            self.module2 = LayerNormMLP(**kwargs)

    def forward(self, observations):
        if self.ensemble:
            return self.module1(observations), self.module2(observations)
        return self.module(observations)


class Representation(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple = (256, 256),
                 activate_final: bool = True, ensemble: bool = True) -> None:
        super().__init__()
        self.ensemble = ensemble
        kwargs = {'input_dim': input_dim, 'hidden_dims': hidden_dims,
                  'activate_final': activate_final, 'activations': nn.GELU}
        self.module = MLP(**kwargs)
        if ensemble:
            self.module1 = MLP(**kwargs)
            self.module2 = MLP(**kwargs)

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
        self.visual = visual
        self.layer_norm = layer_norm
        self.rep_type = rep_type
        self.bottleneck = bottleneck
        self.in_channels = in_channel

        if visual:
            self.v_module = module(in_channels=in_channel)

        rep_cls = LayerNormMLP if layer_norm else MLP
        self.rep = rep_cls(self.input_dim, self.hidden_dims,
                           activate_final=not self.bottleneck, activations=nn.GELU)

    def forward(self, targets, bases=None):
        if bases is None or self.rep_type == 'state':
            inputs = targets
        elif self.rep_type == 'diff':
            inputs = torch.sub(targets, bases) + torch.ones_like(targets) * 1e-6
        elif self.rep_type == 'concat':
            inputs = torch.cat([targets, bases], dim=-1)
        else:
            raise NotImplementedError(f"Unknown rep_type: {self.rep_type}")

        if self.visual:
            inputs = inputs.permute(0, 3, 1, 2)
            inputs = self.v_module(inputs)

        rep = self.rep(inputs)
        if self.bottleneck:
            rep = rep / torch.norm(rep, dim=-1, keepdim=True) * np.sqrt(self.rep_dim)

        return rep


class MonolithicVF(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=(256, 256),
                 readout_size=(256,), use_layer_norm=True, rep_dim=None, obs_rep=0):
        super().__init__()
        repr_cls = LayerNormRepresentation if use_layer_norm else Representation
        self.value_net = repr_cls(input_dim, (*hidden_dims, 1), activate_final=False)

    def forward(self, observations, goals=None, info=False):
        v1, v2 = self.value_net(torch.cat([observations, goals], dim=-1))
        v1, v2 = v1.squeeze(-1), v2.squeeze(-1)
        if info:
            return {'v': (v1 + v2) / 2}
        return v1, v2


def get_rep(encoder: nn.Module, targets: torch.Tensor, bases: torch.Tensor = None):
    if encoder is None:
        return targets
    return encoder(targets) if bases is None else encoder(targets, bases)


class HierarchicalActorCritic(nn.Module):
    def __init__(self, encoders={}, networks={}, use_waypoints=0, lr=3e-4, device="cpu") -> None:
        super().__init__()
        self.device = device
        self.use_waypoints = use_waypoints

        # Store raw dicts for API compatibility; register nn.Module instances properly.
        self.encoders: Dict = encoders
        self.networks: Dict = networks

        # Register nn.Module sub-modules so PyTorch tracks parameters and handles device moves.
        self._encoder_modules = nn.ModuleDict({k: v for k, v in encoders.items() if v is not None})
        self._network_modules = nn.ModuleDict(
            {k: v for k, v in networks.items() if isinstance(v, nn.Module)}
        )

        # Move non-Module networks (e.g. Diffusion_QL) to device manually.
        for v in networks.values():
            if not isinstance(v, nn.Module):
                v.to(device)

        # Move to device (handles all registered nn.Module submodules).
        self.to(device)

        # Build per-component optimizers.
        self.optims = {}
        total_params = 0
        for key, enc in encoders.items():
            if enc is not None:
                self.optims[f"{key}_optim"] = torch.optim.Adam(enc.parameters(), lr=lr)
                total_params += sum(p.numel() for p in enc.parameters() if p.requires_grad)
        for key, net in networks.items():
            if isinstance(net, nn.Module):
                self.optims[f"{key}_optim"] = torch.optim.Adam(net.parameters(), lr=lr)
                total_params += sum(p.numel() for p in net.parameters() if p.requires_grad)

        print(f"Total number of parameters: {total_params / 1e6:.2f}M")

    def value(self, observations, goals, encoded=False, **kwargs):
        state_reps = get_rep(self.encoders['value_state'], targets=observations)
        goal_reps = goals if encoded else get_rep(self.encoders['value_goal'], targets=goals)
        return self.networks['value'](state_reps, goal_reps, **kwargs)

    def target_value(self, observations, goals, **kwargs):
        state_reps = get_rep(self.encoders['value_state'], targets=observations)
        goal_reps = get_rep(self.encoders['value_goal'], targets=goals)
        return self.networks['target_value'](state_reps, goal_reps, **kwargs)

    def actor(self, observations, goals, low_dim_goals=False, state_rep_grad=True,
              goal_rep_grad=True, eval=False, type=None, **kwargs):
        state_reps = get_rep(self.encoders['policy_state'], targets=observations)
        if not state_rep_grad:
            state_reps = state_reps.detach()

        if low_dim_goals:
            goal_reps = goals
        elif self.use_waypoints:
            goal_reps = get_rep(self.encoders['value_goal'], targets=goals)
        else:
            goal_reps = get_rep(self.encoders['policy_goal'], targets=goals, bases=observations)

        if not goal_rep_grad:
            goal_reps = goal_reps.detach()

        cond = torch.cat([state_reps, goal_reps], dim=-1)
        if type is None:
            return self.networks['actor'](cond, **kwargs)
        elif type == 'dql':
            net = self.networks['actor']
            return net.ema_model(cond, **kwargs) if eval else net.actor(cond, **kwargs)

    def high_actor(self, observations, goals, state_rep_grad=True, goal_rep_grad=True, eval=False, **kwargs):
        state_reps = get_rep(self.encoders['high_policy_state'], targets=observations)
        if not state_rep_grad:
            state_reps = state_reps.detach()

        goal_reps = get_rep(self.encoders['high_policy_goal'], targets=goals, bases=observations)
        if not goal_rep_grad:
            goal_reps = goal_reps.detach()

        return self.networks['high_actor'].actor(torch.cat([state_reps, goal_reps], dim=-1))

    def value_goal_encoder(self, targets, bases=None, **kwargs):
        return get_rep(self.encoders['value_goal'], targets=targets)

    def value_state_encoder(self, targets, **kwargs):
        return get_rep(self.encoders['value_state'], targets=targets)

    def high_policy_goal_encoder(self, targets, bases, **kwargs):
        return get_rep(self.encoders['high_policy_goal'], targets=targets, bases=bases)

    def high_policy_state_encoder(self, targets, **kwargs):
        return get_rep(self.encoders['high_policy_state'], targets=targets)

    def policy_state_encoder(self, targets, **kwargs):
        return get_rep(self.encoders['policy_state'], targets=targets)

    def policy_goal_encoder(self, targets, bases, **kwargs):
        assert not self.use_waypoints
        return get_rep(self.encoders['policy_goal'], targets=targets, bases=bases)

    def forward(self, observations, goals):
        return {
            'value': self.value(observations, goals),
            'target_value': self.target_value(observations, goals),
            'actor': self.actor(observations, goals),
            'high_actor': self.high_actor(observations, goals),
        }

    def to(self, device):
        super().to(device)
        # Diffusion_QL is not an nn.Module, so move it manually.
        if 'high_actor' in self.networks and not isinstance(self.networks['high_actor'], nn.Module):
            self.networks['high_actor'].to(device)
        self.device = device
        return self

    def state_dict(self):
        state_dict = {}
        for key, enc in self.encoders.items():
            if enc is not None:
                state_dict[f"{key}_encoder"] = enc.state_dict()
        for key, net in self.networks.items():
            state_dict[f"{key}_network"] = net.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        for key, enc in self.encoders.items():
            if enc is not None:
                enc.load_state_dict(state_dict[f"{key}_encoder"])
        for key, net in self.networks.items():
            net.load_state_dict(state_dict[f"{key}_network"])
