## Sources: 
# https://github.com/ikostrikov/jaxrl2
# https://github.com/henry-prior/jax-rl

from typing import Sequence
from functools import partial

import jax
import numpy as onp
import flax.linen as nn
import jax.numpy as jnp
from jax import random 
from flax.core.frozen_dict import FrozenDict
from haiku import PRNGSequence



def default_init(scale: float = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)


class Encoder(nn.Module):
    features: Sequence[int] = (32, 32, 32, 32)
    filters: Sequence[int] = (3, 3, 3, 3)
    strides: Sequence[int] = (2, 2, 2, 1)
    padding: str = "VALID"
    

    @nn.compact
    def __call__(self, image: jnp.ndarray, proprioception: jnp.ndarray) -> jnp.ndarray:
        assert len(self.features) == len(self.strides)
        # PyTorch uses NCHW while Jax/Flax/TF use NHWC
        x = image.astype(jnp.float32) / 255.0
        # print(image.shape, proprioception.shape)

        for features, filter_, stride in zip(self.features, self.filters, self.strides):
            x = nn.Conv(
                features,
                kernel_size=(filter_, filter_),
                strides=(stride, stride),
                kernel_init=default_init(),
                padding=self.padding,
            )(x)
            # print("-->", x.shape)
            x = nn.relu(x)
            
        x = x.reshape((*x.shape[:-3], -1))
        x = nn.Dense(50, kernel_init=default_init())(x)
        x = nn.LayerNorm()(x)
        x = nn.tanh(x)
        x = jnp.concatenate([x, proprioception], axis=-1)
        return x
    

class DoubleCritic(nn.Module):
    encoder: nn.Module

    @nn.compact
    def __call__(self, image: jnp.ndarray, proprioception: jnp.ndarray, action, Q1=False):
        
        x = self.encoder(image, proprioception)
        state_action = jnp.concatenate([x, action], axis=-1)

        q1 = nn.Dense(features=500)(state_action)
        q1 = nn.LayerNorm()(q1)
        q1 = nn.tanh(q1)
        q1 = nn.Dense(features=500)(q1)
        q1 = nn.elu(q1)
        q1 = nn.Dense(features=1)(q1)

        if Q1:
            return q1

        q2 = nn.Dense(features=500)(state_action)
        q2 = nn.LayerNorm()(q2)
        q2 = nn.tanh(q2)
        q2 = nn.Dense(features=500)(q2)
        q2 = nn.elu(q2)
        q2 = nn.Dense(features=1)(q2)

        return q1, q2
    

def build_double_critic_model(input_shapes, init_rng):
    init_batch = [jnp.ones(shape, jnp.float32) for shape in input_shapes]
    critic = DoubleCritic(Encoder())
    init_variables = critic.init(init_rng, *init_batch)

    return init_variables["params"]


@partial(jax.jit, static_argnums=4)
def apply_double_critic_model(
    params: FrozenDict, image: jnp.ndarray, proprioception: jnp.ndarray, action: jnp.ndarray, Q1: bool,
) -> jnp.ndarray:
    return DoubleCritic(Encoder()).apply(dict(params=params), image, proprioception, action, Q1=Q1)




@jax.jit
@jax.vmap
def gaussian_likelihood(
    sample: jnp.ndarray, mu: jnp.ndarray, log_sig: jnp.ndarray
) -> jnp.ndarray:
    """
    Calculates the log likelihood of a sample from a Gaussian distribution.
    i.e. the log of the pdf evaluated at `sample`
    Args:
        sample (jnp.ndarray): an array of samples from the distribution
        mu (jnp.ndarray): the mean of the distribution
        log_sig (jnp.ndarray): the log of the standard deviation of the distribution
    Returns:
        the log likelihood of the sample
    """
    return -0.5 * (
        ((sample - mu) / (jnp.exp(log_sig) + 1e-6)) ** 2
        + 2 * log_sig
        + jnp.log(2 * onp.pi)
    )



class GaussianPolicy(nn.Module):
    action_dim: int
    max_action: float
    encoder: nn.Module
    log_sig_min: float = -20.0
    log_sig_max: float = None

    @nn.compact
    def __call__(self, image: jnp.ndarray, proprioception: jnp.ndarray, key=None, sample=False, MPO=False):
        # print(image.shape, proprioception.shape)
        x = self.encoder(image, proprioception)
        # print(image.shape, proprioception.shape, x.shape, self.action_dim)
        x = nn.Dense(features=200)(x)
        x = nn.LayerNorm()(x)
        x = nn.tanh(x)
        x = nn.Dense(features=200)(x)
        x = nn.elu(x)
        x = nn.Dense(features=2 * self.action_dim)(x)

        mu, log_sig = jnp.split(x, 2, axis=-1)
        log_sig = jnp.clip(log_sig, self.log_sig_min, self.log_sig_max)

        if MPO:
            return mu, log_sig

        if not sample:
            return self.max_action * nn.tanh(mu), log_sig
        else:
            sig = jnp.exp(log_sig)
            pi = mu + random.normal(key, mu.shape) * sig
            log_pi = gaussian_likelihood(pi, mu, log_sig)
            pi = nn.tanh(pi)
            log_pi -= jnp.sum(
                jnp.log(nn.relu(1 - pi ** 2) + 1e-6), axis=1, keepdims=True,
            )
            return self.max_action * pi, log_pi


def build_gaussian_policy_model(input_shapes, action_dim, max_action, init_rng):
    init_batch = [jnp.ones(shape, jnp.float32) for shape in input_shapes]
    # print(input_shapes, action_dim, max_action, init_rng, latent_dim)
    policy = GaussianPolicy(action_dim=action_dim, max_action=max_action, encoder=Encoder())
    init_variables = policy.init(init_rng, *init_batch)

    return init_variables["params"]


@partial(jax.jit, static_argnums=(1, 2, 6, 7))
def apply_gaussian_policy_model(
    params: FrozenDict,
    action_dim: int,  #
    max_action: float, #
    image: jnp.ndarray,
    proprioception: jnp.ndarray,
    key: PRNGSequence,
    sample: bool, #
    MPO: bool, #
) -> jnp.ndarray:
    return GaussianPolicy(action_dim=action_dim, max_action=max_action, encoder=Encoder()).apply(
        dict(params=params), image, proprioception, key=key, sample=sample, MPO=MPO
    )





class Constant(nn.Module):
    start_value: float
    absolute: bool = False

    @nn.compact
    def __call__(self, dtype=jnp.float32):
        value = self.param(
            "value", lambda key, shape: jnp.full(shape, self.start_value, dtype), (1,)
        )
        if self.absolute:
            value = nn.softplus(value)
        return jnp.asarray(value, dtype)


def build_constant_model(
    start_value: float, init_rng: PRNGSequence, absolute: bool = False
) -> FrozenDict:
    constant = Constant(start_value=start_value, absolute=absolute)
    init_variables = constant.init(init_rng)

    return init_variables["params"]


@partial(jax.jit, static_argnums=(1, 2))
def apply_constant_model(
    params: FrozenDict, start_value: float, absolute: bool,
) -> jnp.ndarray:
    return Constant(start_value=start_value, absolute=absolute).apply(
        dict(params=params)
    )




# key = random.PRNGKey(0) 
# g = build_gaussian_policy_model([(1, 90, 160, 3), (1, 10)], 3, 1.0, key, 50)
# i = build_double_critic_model([(1, 90, 160, 3), (1, 10), (1, 5)], key, 50)