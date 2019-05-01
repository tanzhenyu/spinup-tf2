import numpy as np
import tensorflow as tf
import scipy.signal
from gym.spaces import Box, Discrete

EPS = 1e-8

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(ob_space, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    model = tf.keras.Sequential()
    for h in hidden_sizes[:-1]:
        model.add(tf.keras.layers.Dense(units=h, activation=activation))
    model.add(tf.keras.layers.Dense(units=hidden_sizes[-1], activation=output_activation))
    model.build(input_shape=(None,) + ob_space.shape)
    return model

def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


"""
Policies
"""
class Mlp_Categorical_Actor_Critic(tf.keras.Model):

  def __init__(self, ob_space, ac_space, hidden_sizes=(64, 64), activation=tf.keras.activations.tanh, output_activation=None):
    super(Mlp_Categorical_Actor_Critic, self).__init__()
    self.act_dim = ac_space.n
    with tf.name_scope('pi'):
      self.actor_mlp = mlp(ob_space=ob_space, hidden_sizes=list(hidden_sizes)+[self.act_dim], activation=activation)
    with tf.name_scope('v'):
      self.critic_mlp = mlp(ob_space=ob_space, hidden_sizes=list(hidden_sizes)+[1], activation=activation)

  @tf.function
  def get_pi_logpi_vf(self, observations):
    logits = self.actor_mlp(observations)
    logp_all = tf.nn.log_softmax(logits)
    pi = tf.squeeze(tf.random.categorical(logits, num_samples=1, seed=0), axis=1)
    logp_pi = tf.reduce_sum(tf.one_hot(pi, depth=self.act_dim) * logp_all, axis=1)
    vf = self.critic_mlp(observations)
    return pi, logp_pi, vf

  @tf.function
  def get_logp(self, observations, actions):
    logits = self.actor_mlp(observations)
    logp_all = tf.nn.log_softmax(logits)
    return tf.reduce_sum(tf.one_hot(actions, depth=self.act_dim) * logp_all, axis=1)

  @tf.function
  def get_v(self, observations):
    return tf.squeeze(self.critic_mlp(observations), axis=1)

class Mlp_Gaussian_Actor_Critic(tf.keras.Model):

  def __init__(self, ob_space, ac_space, hidden_sizes, activation, output_activation):
    super(Mlp_Gaussian_Actor_Critic, self).__init__()
    self.act_dim=ac_space.shape[-1]
    with tf.name_scope('pi'):
      self.actor_mlp = mlp(ob_space=ob_space, hidden_sizes=list(hidden_sizes)+[self.act_dim], activation=activation)
    with tf.name_scope('v'):
      self.critic_mlp = mlp(ob_space=ob_space, hidden_sizes=list(hidden_sizes)+[1], activation=activation)
      self.critic_mlp.log_std = self.add_weight(name='log_std', shape=(self.act_dim,), initializer=tf.initializers.constant(-0.5))
      self.log_std = self.critic_mlp.log_std

  @tf.function
  def get_pi_logpi_vf(self, observations):
    mu = self.actor_mlp(observations)
    std = tf.exp(self.log_std)
    pi = mu + tf.random.normal(tf.shape(mu)) * std
    logp_pi = gaussian_likelihood(pi, mu, self.log_std)
    vf = self.critic_mlp(observations)
    return pi, logp_pi, vf

  @tf.function
  def get_logp(self, observations, actions):
    mu = self.actor_mlp(observations)
    return gaussian_likelihood(actions, mu, self.log_std)

  @tf.function
  def get_v(self, observations):
    return tf.squeeze(self.critic_mlp(observations), axis=1)


def mlp_actor_critic(ob_space, ac_space, hidden_sizes=(64,64), activation=tf.tanh, output_activation=None):
  if isinstance(ac_space, Box):
    model = Mlp_Gaussian_Actor_Critic(ob_space, ac_space, hidden_sizes, activation, output_activation)
  elif isinstance(ac_space, Discrete):
    model = Mlp_Categorical_Actor_Critic(ob_space, ac_space, hidden_sizes, activation, output_activation)
  return model 
