"""Common networks used in RL.

This file contains nn.Module definitions for common networks used in RL. It is divided into three sets:

1) Common Networks: MLP
2) Common RL Networks:
    For discrete action spaces: DiscreteCritic is a Q-function
    For continuous action spaces: Critic, ValueCritic, and Policy provide the Q-function, value function, and policy respectively.
    For ensembling: ensemblize() provides a wrapper for creating ensembles of networks (e.g. for min-Q / double-Q)
3) Meta Networks for vision tasks:
    WithEncoder: Combines a fully connected network with an encoder network (encoder may come from rl_m.vision)
    ActorCritic: Same as WithEncoder, but for possibly many different networks (e.g. actor, critic, value)
"""

from rl_m.typing import *
import torch
from torch import nn
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.independent import Independent
import distrax
import numpy as np

###############################
#
#  Common Networks
#
###############################


from typing import Optional, Sequence, Callable
import torch
from torch import nn
import numpy as np


def default_init(weight, scale: Optional[float] = 1.0):
    if isinstance(weight, nn.Linear):
        torch.nn.init.xavier_uniform_(weight.weight, gain=scale)
        torch.nn.init.constant_(weight.bias, 0)


class MLP(nn.Module):
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: Sequence[int],
                 activations: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU,
                 activate_final=False,
                 kernel_init: Callable[[torch.Tensor], torch.Tensor] = default_init,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
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

    def forward(self, x):

        return self.model(x)

###############################
#
#
#  Common RL Networks
#
###############################

class DiscreteCritic(nn.Module):
    def __init__(self, hidden_dims: Sequence[int], n_actions: int, activations: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.n_actions = n_actions
        self.activations = activations
        self.model = MLP((*self.hidden_dims, self.n_actions), activations=self.activations)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.model(observations)

class Critic(nn.Module):
    def __init__(self, hidden_dims: Sequence[int], activations: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.activations = activations
        self.model = MLP((*self.hidden_dims, 1), activations=self.activations)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([observations, actions], dim=-1)
        critic = self.model(inputs)
        return torch.squeeze(critic, -1)



def ensemblize(cls, num_qs, out_axes=0, **kwargs):
    """
    Useful for making ensembles of Q functions (e.g. double Q in SAC).

    Usage:

        critic_def = ensemblize(Critic, 2)(hidden_dims=hidden_dims)

    """
    return nn.ModuleList([cls(**kwargs) for _ in range(num_qs)])


class ValueCritic(nn.Module):
    def __init__(self, hidden_dims):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1)
        )

    def forward(self, observations):
        critic = self.critic(observations)
        return torch.squeeze(critic, -1)


class Policy(nn.Module):
    def __init__(self, obs_dim, hidden_dims, action_dim, log_std_min=-5, log_std_max=2, tanh_squash_distribution=False, state_dependent_std=True, final_fc_init_scale=1e-2):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dims = hidden_dims
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.tanh_squash_distribution = tanh_squash_distribution
        self.state_dependent_std = state_dependent_std
        self.final_fc_init_scale = final_fc_init_scale

        model_list = nn.ModuleList()
        model_list.append(nn.Linear(obs_dim, hidden_dims[0]))
        model_list.append(nn.ReLU())
        for i in range(1, len(hidden_dims)):
            model_list.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            model_list.append(nn.ReLU())

        self.outputs = nn.Sequential(*model_list)

        self.means = nn.Linear(hidden_dims[-1], action_dim)
        if self.state_dependent_std:
            self.log_stds = nn.Linear(hidden_dims[-1], action_dim)
        else:
            self.log_stds = nn.Parameter(torch.zeros(action_dim,))
        
        self.apply(default_init)

    def forward(self, observations, temperature=1.0):

        outputs = self.outputs(observations)

        means = self.means(outputs)
        if self.state_dependent_std:
            log_stds = self.log_stds(outputs)
        else:
            log_stds = self.log_stds

        log_stds = torch.clamp(log_stds, self.log_std_min, self.log_std_max)
        distribution = Independent(
            Normal(loc=means, scale=torch.exp(log_stds)*np.maximum(1e-6, temperature).item()), 1
        )

        if self.tanh_squash_distribution:
            distribution = torch.distributions.transformed_distribution.TransformedDistribution(
                distribution, torch.distributions.transforms.TanhTransform()
            )

        return distribution


class DiscretePolicy(nn.Module):
    def __init__(self, input_dims, hidden_dims, action_dim, final_fc_init_scale=1e-2):
        super().__init__()
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.action_dim = action_dim
        self.final_fc_init_scale = final_fc_init_scale

        self.outputs = nn.Sequential(
            nn.Linear(input_dims, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU()
        )

        self.logits = nn.Linear(hidden_dims[1], action_dim)

    def forward(self, observations, temperature=1.0):
        outputs = self.outputs(observations)

        logits = self.logits(outputs)

        distribution = torch.distributions.Categorical(logits=logits)

        return distribution


class TransformedWithMode(distrax.Transformed):
    def mode(self) -> np.ndarray:
        return self.bijector.forward(self.distribution.mode())


###############################
#
#
#   Meta Networks for Encoders
#
###############################


def get_latent(
    encoder: nn.Module, observations: Union[np.ndarray, Dict[str, np.ndarray]]
):
    """

    Get latent representation from encoder. If observations is a dict
        a state and image component, then concatenate the latents.

    """
    if encoder is None:
        return observations

    elif isinstance(observations, dict):
        return torch.cat(
            [encoder(observations["image"]), observations["state"]], axis=-1
        )

    else:
        return encoder(observations)


class WithEncoder(nn.Module):
    def __init__(self, encoder, network):
        super().__init__()
        self.encoder = encoder
        self.network = network

    def forward(self, observations, *args, **kwargs):
        latents = get_latent(self.encoder, observations)
        return self.network(latents, *args, **kwargs)


class ActorCritic(nn.Module):
    """Combines FC networks with encoders for actor, critic, and value.

    Note: You can share encoder parameters between actor and critic by passing in the same encoder definition for both.

    Example:

        encoder_def = ImpalaEncoder()
        actor_def = Policy(...)
        critic_def = Critic(...)
        # This will share the encoder between actor and critic
        model_def = ActorCritic(
            encoders={'actor': encoder_def, 'critic': encoder_def},
            networks={'actor': actor_def, 'critic': critic_def}
        )
        # This will have separate encoders for actor and critic
        model_def = ActorCritic(
            encoders={'actor': encoder_def, 'critic': copy.deepcopy(encoder_def)},
            networks={'actor': actor_def, 'critic': critic_def}
        )
    """

    def __init__(self, encoders, networks):
        super().__init__()
        self.encoders = nn.ModuleDict(encoders)
        self.networks = nn.ModuleDict(networks)

    def actor(self, observations, **kwargs):
        latents = get_latent(self.encoders["actor"], observations)
        return self.networks["actor"](latents, **kwargs)

    def critic(self, observations, actions, **kwargs):
        latents = get_latent(self.encoders["critic"], observations)
        return self.networks["critic"](latents, actions, **kwargs)

    def value(self, observations, **kwargs):
        latents = get_latent(self.encoders["value"], observations)
        return self.networks["value"](latents, **kwargs)

    def forward(self, observations, actions):
        rets = {}
        if "actor" in self.networks:
            rets["actor"] = self.actor(observations)
        if "critic" in self.networks:
            rets["critic"] = self.critic(observations, actions)
        if "value" in self.networks:
            rets["value"] = self.value(observations)
        return rets
