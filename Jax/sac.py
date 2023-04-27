## Source
# https://github.com/henry-prior/jax-rl/blob/master/jax_rl/SAC.py

from functools import partial
from typing import Tuple
import multiprocessing 
from multiprocessing import Process
from multiprocessing import Queue
from time import time



import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import optim
from flax.core.frozen_dict import FrozenDict
from haiku import PRNGSequence

from Common.ReplayBuffer import ReplayBuffer
from models import apply_constant_model
from models import apply_double_critic_model
from models import apply_gaussian_policy_model
from models import build_constant_model
from models import build_double_critic_model
from models import build_gaussian_policy_model
# from jax_rl.saving import load_model
# from jax_rl.saving import save_model
# from jax_rl.utils import copy_params
# from jax_rl.utils import double_mse


@jax.vmap
def double_mse(q1: jnp.ndarray, q2: jnp.ndarray, qt: jnp.ndarray) -> float:
    return jnp.square(qt - q1).mean() + jnp.square(qt - q2).mean()

def actor_loss_fn(log_alpha: jnp.ndarray, log_p: jnp.ndarray, min_q: jnp.ndarray):
    return (jnp.exp(log_alpha) * log_p - min_q).mean()


def alpha_loss_fn(log_alpha: jnp.ndarray, target_entropy: float, log_p: jnp.ndarray):
    return (log_alpha * (-log_p - target_entropy)).mean()

@jax.jit
def copy_params(
    orig_params: FrozenDict, target_params: FrozenDict, tau: float
) -> nn.Module:
    """
    Applies polyak averaging between two sets of parameters.
    """
    update_params = jax.tree_map(
        lambda m1, mt: tau * m1 + (1 - tau) * mt, orig_params, target_params,
    )

    return update_params


@partial(jax.jit, static_argnums=(5, 6, 7))
def get_td_target(
    rng: PRNGSequence,
    next_image: jnp.ndarray,
    next_proprioception: jnp.ndarray,
    reward: jnp.ndarray,
    not_done: jnp.ndarray,
    discount: float,
    max_action: float,
    action_dim: int,
    actor_params: FrozenDict,
    critic_target_params: FrozenDict,
    log_alpha_params: FrozenDict,
) -> jnp.ndarray:
    next_action, next_log_p = apply_gaussian_policy_model(
        actor_params, action_dim, max_action, next_image, next_proprioception, rng, True, False
    )

    target_Q1, target_Q2 = apply_double_critic_model(
        critic_target_params, next_image, next_proprioception, next_action, False
    )
    target_Q = (
        jnp.minimum(target_Q1, target_Q2)
        - jnp.exp(apply_constant_model(log_alpha_params, -3.5, False)) * next_log_p
    )
    target_Q = reward + not_done * discount * target_Q

    return target_Q


@jax.jit
def critic_step(
    optimizer: optim.Optimizer,
    image: jnp.ndarray,
    proprioception: jnp.ndarray,
    action: jnp.ndarray,
    target_Q: jnp.ndarray,
) -> optim.Optimizer:
    def loss_fn(critic_params):
        current_Q1, current_Q2 = apply_double_critic_model(
            critic_params, image, proprioception, action, False
        )
        critic_loss = double_mse(current_Q1, current_Q2, target_Q)
        return jnp.mean(critic_loss)

    grad = jax.grad(loss_fn)(optimizer.target)
    return optimizer.apply_gradient(grad)


@partial(jax.jit, static_argnums=(6, 7))
def actor_step(
    rng: PRNGSequence,
    optimizer: optim.Optimizer,
    critic_params: FrozenDict,
    image: jnp.ndarray,
    proprioception: jnp.ndarray,
    log_alpha_params: FrozenDict,
    max_action: float,
    action_dim: int,
) -> Tuple[optim.Optimizer, jnp.ndarray]:
    def loss_fn(actor_params):
        actor_action, log_p = apply_gaussian_policy_model(
            actor_params, action_dim, max_action, image, proprioception, rng, True, False
        )
        q1, q2 = apply_double_critic_model(critic_params, image, proprioception, actor_action, False)
        min_q = jnp.minimum(q1, q2)
        partial_loss_fn = jax.vmap(
            partial(
                actor_loss_fn,
                jax.lax.stop_gradient(
                    apply_constant_model(log_alpha_params, -3.5, False)
                ),
            ),
        )
        actor_loss = partial_loss_fn(log_p, min_q)
        return jnp.mean(actor_loss), log_p

    grad, log_p = jax.grad(loss_fn, has_aux=True)(optimizer.target)
    return optimizer.apply_gradient(grad), log_p


@partial(jax.jit, static_argnums=2)
def alpha_step(
    optimizer: optim.Optimizer, log_p: jnp.ndarray, target_entropy: float
) -> optim.Optimizer:
    log_p = jax.lax.stop_gradient(log_p)

    def loss_fn(log_alpha_params):
        partial_loss_fn = jax.vmap(
            partial(
                alpha_loss_fn,
                apply_constant_model(log_alpha_params, -3.5, False),
                target_entropy,
            )
        )
        return jnp.mean(partial_loss_fn(log_p))

    grad = jax.grad(loss_fn)(optimizer.target)
    return optimizer.apply_gradient(grad)


class SAC:
    def __init__(self, args):
        self._args = args
        
        self.rng = PRNGSequence(self._args.seed)  

        actor_params = build_gaussian_policy_model(
            self._args.actor_input_dim, self._args.action_dim, self._args.max_action, next(self.rng)
        )
        actor_optimizer = optim.Adam(learning_rate=self._args.actor_lr).create(actor_params)
        self.actor_optimizer = jax.device_put(actor_optimizer)

        init_rng = next(self.rng)

        critic_params = build_double_critic_model(self._args.critic_input_dim, init_rng)
        self.critic_target_params = build_double_critic_model(self._args.critic_input_dim, init_rng)
        critic_optimizer = optim.Adam(learning_rate=self._args.critic_lr).create(critic_params)
        self.critic_optimizer = jax.device_put(critic_optimizer) 

        log_alpha_params = build_constant_model(-3.5, next(self.rng))
        log_alpha_optimizer = optim.Adam(learning_rate=self._args.critic_lr).create(log_alpha_params)
        self.log_alpha_optimizer = jax.device_put(log_alpha_optimizer)
        self.target_entropy = -self._args.action_dim

        self.max_action = self._args.max_action
        self.discount = self._args.discount
        self.tau = self._args.critic_tau 

        self.action_dim = self._args.action_dim

        self.replay_buffer = ReplayBuffer(
            image_shape=self._args.image_shape,
            proprioception_shape=self._args.proprioception_shape,
            action_shape=(self._args.action_dim,),
            capacity=self._args.replay_buffer_capacity,
            batch_size=self._args.batch_size) 


    @property
    def target_params(self):
        return (
            self.discount,
            self.max_action,
            self.action_dim,
            self.actor_optimizer.target,
            self.critic_target_params,
            self.log_alpha_optimizer.target,
        )

    def select_action(self, image: jnp.ndarray, proprioception: jnp.ndarray) -> jnp.ndarray:
        image = jnp.expand_dims(image, axis=0)
        proprioception = jnp.expand_dims(proprioception, axis=0)
        # print(">>", image.shape, proprioception.shape)
        mu, _ = apply_gaussian_policy_model(
            self.actor_optimizer.target,
            self.action_dim,
            self.max_action,
            image,
            proprioception,
            None,
            False,
            False,
        )
        return mu.flatten()

    def sample_action(self, rng: PRNGSequence, image: jnp.ndarray, proprioception: jnp.ndarray) -> jnp.ndarray:
        mu, log_sig = apply_gaussian_policy_model(
            self.actor_optimizer.target,
            self.action_dim,
            self.max_action,
            image,
            proprioception,
            None,
            False,
            False,
        )
        return mu + jax.random.normal(rng, mu.shape) * jnp.exp(log_sig)

    def train(self, images, propris, actions, rewards, next_images, next_propris, not_dones):
        images = jax.device_put(images)
        propris = jax.device_put(propris)
        actions = jax.device_put(actions)
        rewards = jax.device_put(rewards)
        next_images = jax.device_put(next_images)
        next_propris = jax.device_put(next_propris)
        not_dones = jax.device_put(not_dones) ## !!!

        target_Q = jax.lax.stop_gradient(
            get_td_target(next(self.rng), next_images, next_propris, rewards, not_dones, *self.target_params)
        )

        self.critic_optimizer = critic_step(
            self.critic_optimizer, images, propris, actions, target_Q
        )

        self.actor_optimizer, log_p = actor_step(
            next(self.rng),
            self.actor_optimizer,
            self.critic_optimizer.target,
            images,
            propris,
            self.log_alpha_optimizer.target,
            self.max_action,
            self.action_dim,
        )

        
        self.log_alpha_optimizer = alpha_step(
            self.log_alpha_optimizer, log_p, self.target_entropy
        )

        self.critic_target_params = copy_params(
            self.critic_target_params, self.critic_optimizer.target, self.tau
        )


    # def save(self, filename):
    #     save_model(filename + "_critic", self.critic_optimizer)
    #     save_model(filename + "_actor", self.actor_optimizer)

    # def load(self, filename):
    #     self.critic_optimizer = load_model(filename + "_critic", self.critic_optimizer)
    #     self.critic_optimizer = jax.device_put(self.critic_optimizer)
    #     self.critic_target_params = self.critic_optimizer.target.copy()

    #     self.actor_optimizer = load_model(filename + "_actor", self.actor_optimizer)
    #     self.actor_optimizer = jax.device_put(self.actor_optimizer)

    # export XLA_PYTHON_CLIENT_PREALLOCATE=false