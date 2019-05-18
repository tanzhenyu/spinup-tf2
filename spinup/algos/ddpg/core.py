import numpy as np
import tensorflow as tf


def mlp(input_shape, hidden_sizes=(32,), activation='tanh', output_activation=None):
    model = tf.keras.Sequential()
    for h in hidden_sizes[:-1]:
        model.add(tf.keras.layers.Dense(units=h, activation=activation))
    model.add(tf.keras.layers.Dense(units=hidden_sizes[-1], activation=output_activation))
    model.build(input_shape=(None,) + input_shape)
    return model

"""
Actor-Critics
"""
def mlp_actor_critic(obs_dim, act_dim, hidden_sizes=(400,300), activation='relu', 
                     output_activation='tanh'):
    with tf.name_scope('pi'):
        pi_network = mlp((obs_dim,), list(hidden_sizes)+[act_dim], activation, output_activation)
    with tf.name_scope('q'):
        q_network = mlp((obs_dim+act_dim,), list(hidden_sizes)+[1], activation, None)
    return pi_network, q_network