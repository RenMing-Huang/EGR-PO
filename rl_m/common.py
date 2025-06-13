from rl_m.typing import *
import torch
from torch import nn
import numpy as np
import functools

import torch
import functools

nonpytree_field = functools.partial(torch.nn.Parameter, requires_grad=False)


def shard_batch(batch):
    d = torch.cuda.device_count()

    def reshape(x):
        assert (
            x.shape[0] % d == 0
        ), f"Batch size needs to be divisible by # devices, got {x.shape[0]} and {d}"
        return x.reshape((d, x.shape[0] // d, *x.shape[1:]))

    return torch.utils.data.DataLoader(batch, batch_size=batch.shape[0] // d)


def target_update(
    model, target_model, tau: float
):
    for param, target_param in zip(model.parameters(), target_model.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


import torch

class TrainState:
    """
    Core abstraction of a model in this repository.

    Creation:
    ```
        model_def = nn.Linear(4, 12) # or any other PyTorch Module
        params = model_def.state_dict()
        model = TrainState.create(model_def, params, optimizer=None) # Optionally, pass in a PyTorch optimizer
    ```

    Usage:
    ```
        y = model(torch.ones((1, 4))) # By default, uses the `forward` method of the model_def and params stored in TrainState
        y = model(torch.ones((1, 4)), params=params) # You can pass in params (useful for gradient computation)
        y = model(torch.ones((1, 4)), method=method) # You can apply a different method as well
    ```

    More complete example:
    ```
        def loss(params):
            y_pred = model(x, params=params)
            return torch.mean((y - y_pred) ** 2)

        grads = torch.autograd.grad(loss, model.params.values())
        new_model = model.apply_gradients(grads=grads) # Alternatively, new_model = model.apply_loss_fn(loss_fn=loss)
    ```
    """

    def __init__(
        self,
        step: int,
        apply_fn: Callable[..., Any],
        model_def: Any,
        params: dict,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        self.step = step
        self.apply_fn = apply_fn
        self.model_def = model_def
        self.params = params
        self.optimizer = optimizer

    @classmethod
    def create(
        cls,
        model_def: nn.Module,
        params: dict,
        optimizer: Optional[torch.optim.Optimizer] = None,
        **kwargs,
    ) -> "TrainState":
        return cls(
            step=1,
            apply_fn=model_def,
            model_def=model_def,
            params=params,
            optimizer=optimizer,
            **kwargs,
        )

    def __call__(
        self,
        *args,
        params=None,
        extra_variables: dict = None,
        method: str = None,
        **kwargs,
    ):
        """
        Internally calls model_def.forward with the following logic:

        Arguments:
            params: If not None, use these params instead of the ones stored in the model.
            extra_variables: Additional variables to pass into forward
            method: If None, use the `forward` method of the model_def. If a string, uses
                the method of the model_def with that name (e.g. 'encode' -> model_def.encode).
                If a function, uses that function.

        """
        if params is None:
            params = self.params

        if extra_variables is not None:
            variables = {**params, **extra_variables}
        else:
            variables = params

        if method is None:
            method = 'forward'

        if isinstance(method, str):
            method = getattr(self.model_def, method)

        return method(*args, **variables, **kwargs)

    def apply_gradients(self, *, grads, **kwargs):
        """Updates `step`, `params`, `optimizer` and `**kwargs` in return value.

        Note that internally this function calls `optimizer.step()` to update `params`.

        Args:
            grads: Gradients that have the same structure as `.params`.
            **kwargs: Additional attributes that should be updated.

        Returns:
            An updated instance of `self` with `step` incremented by one, `params`
            and `optimizer` updated by applying `grads`, and additional attributes
            updated as specified by `kwargs`.
        """
        for i, (name, param) in enumerate(self.params.items()):
            self.params[name] = param - self.optimizer.param_groups[0]['lr'] * grads[i]

        self.optimizer.step()

        return self.__class__(
            step=self.step + 1,
            apply_fn=self.apply_fn,
            model_def=self.model_def,
            params=self.params,
            optimizer=self.optimizer,
            **kwargs,
        )

    def apply_loss_fn(self, *, loss_fn, pmap_axis=None, has_aux=False):
        """
        Takes a gradient step towards minimizing `loss_fn`. Internally, this calls
        `torch.autograd.grad` followed by `TrainState.apply_gradients`. If pmap_axis is provided,
        additionally it averages gradients (and info) across devices before performing update.
        """
        if has_aux:
            loss, info = loss_fn(self.params)
            grads = torch.autograd.grad(loss, self.params.values())
            if pmap_axis is not None:
                grads = torch.mean(grads, dim=pmap_axis)
                info = torch.mean(info, dim=pmap_axis)

            return self.apply_gradients(grads=grads), info

        else:
            loss = loss_fn(self.params)
            grads = torch.autograd.grad(loss, self.params.values())
            if pmap_axis is not None:
                grads = torch.mean(grads, dim=pmap_axis)
            return self.apply_gradients(grads=grads)
