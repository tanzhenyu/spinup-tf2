import numpy as np
import tensorflow as tf
import gym
import time
import spinup.algos.ppo.core as core
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_tf import MpiAdamOptimizer, sync_params
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, ob_space, ac_space, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, ob_space.shape), dtype=ob_space.dtype)
        self.act_buf = np.zeros(core.combined_shape(size, ac_space.shape), dtype=ac_space.dtype)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        return [self.obs_buf, self.act_buf, self.adv_buf, 
                self.ret_buf, self.logp_buf]


"""

Proximal Policy Optimization (by clipping), 

with early stopping based on approximate KL

"""
def ppo(env_fn, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
        target_kl=0.01, logger_kwargs=dict(), save_freq=10):
    """

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: A function which takes in placeholder symbols 
            for state, ``x_ph``, and action, ``a_ph``, and returns the main 
            outputs from the agent's Tensorflow computation graph:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       (batch, act_dim)  | Samples actions from policy given 
                                           | states.
            ``logp``     (batch,)          | Gives log probability, according to
                                           | the policy, of taking actions ``a_ph``
                                           | in states ``x_ph``.
            ``logp_pi``  (batch,)          | Gives log probability, according to
                                           | the policy, of the action sampled by
                                           | ``pi``.
            ``v``        (batch,)          | Gives the value estimate for states
                                           | in ``x_ph``. (Critical: make sure 
                                           | to flatten this!)
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the actor_critic 
            function you provided to PPO.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        clip_ratio (float): Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while 
            still profiting (improving the objective function)? The new policy 
            can still go farther than the clip_ratio says, but it doesn't help
            on the objective anymore. (Usually small, 0.1 to 0.3.)

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.

        train_pi_iters (int): Maximum number of gradient descent steps to take 
            on policy loss per epoch. (Early stopping may cause optimizer
            to take fewer than this.)

        train_v_iters (int): Number of gradient descent steps to take on 
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used 
            for early stopping. (Usually small, 0.01 or 0.05.)

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    seed += 10000 * proc_id()
    tf.random.set_seed(seed)
    np.random.seed(seed)

    env = env_fn()
    ob_space = env.observation_space
    ac_space = env.action_space
    obs_dim = ob_space.shape
    act_dim = ac_space.shape

    model = actor_critic(ob_space, ac_space)
    
    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space

    # Optimizers
    opt_pi = tf.optimizers.Adam(learning_rate=pi_lr)
    opt_v = tf.optimizers.Adam(learning_rate=vf_lr)

    # Experience buffer
    local_steps_per_epoch = int(steps_per_epoch)
    buf = PPOBuffer(ob_space, ac_space, local_steps_per_epoch, gamma, lam)

    actor_weights = model.actor_mlp.trainable_weights
    critic_weights = model.critic_mlp.trainable_weights

    # Setup model saving
    # logger.setup_tf_saver(sess, inputs={'x': x_ph}, outputs={'pi': pi, 'v': v})

    @tf.function
    def update(obs, acs, advs, rets, logp_olds):
      
      def pi_loss_fn():
        logp = model.get_logp(obs, acs)
        ratio = tf.exp(logp - logp_olds)
        min_adv = tf.where(advs > 0, (1+clip_ratio)*advs, (1-clip_ratio)*advs)
        pi_loss = -tf.reduce_mean(tf.minimum(ratio * advs, min_adv))
        return pi_loss

      def v_loss_fn():
        v = model.get_v(obs)
        v_loss = tf.reduce_mean((rets - v)**2)
        return v_loss

      def approx_kl_fn():
        logp = model.get_logp(obs, acs)
        approx_kl = tf.reduce_mean(logp_olds - logp)
        return approx_kl

      def approx_ent_fn():
        logp = model.get_logp(obs, acs)
        return tf.reduce_mean(-logp)

      pi_loss_old = pi_loss_fn()
      v_loss_old = v_loss_fn()
      ent = approx_ent_fn()

      def train_pi():
        with tf.GradientTape() as tape:
          logp = model.get_logp(obs, acs)
          ratio = tf.exp(logp - logp_olds)
          min_adv = tf.where(advs > 0, (1+clip_ratio)*advs, (1-clip_ratio)*advs)
          pi_loss = -tf.reduce_mean(tf.minimum(ratio * advs, min_adv))
        grads = tape.gradient(pi_loss, actor_weights)
        opt_pi.apply_gradients(zip(grads, actor_weights))
        kl = tf.reduce_mean(logp_olds - logp)
        return kl

      stopIter = tf.constant(train_pi_iters)
      for i in tf.range(train_pi_iters):
        kl = train_pi()
        if kl > 1.5 * target_kl:
          stopIter = i
          break

      for i in tf.range(train_v_iters):
        opt_v.minimize(v_loss_fn, critic_weights)

      pi_loss_new = pi_loss_fn()
      v_loss_new = v_loss_fn()
      kl = approx_kl_fn()

      return pi_loss_old, v_loss_old, kl, ent, pi_loss_new-pi_loss_old, v_loss_new-v_loss_old, stopIter


    start_time = time.time()
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        for t in range(local_steps_per_epoch):
            expand_o = tf.constant(o.reshape(1, -1))
            a, logp_t, v_t = model.get_pi_logpi_vf(expand_o)

            a = a.numpy()[0]
            logp_t = logp_t.numpy()[0]
            v_t = v_t.numpy()[0][0]
            buf.store(o, a, r, v_t, logp_t)
            logger.store(VVals=v_t)

            o, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1

            terminal = d or (ep_len == max_ep_len)
            if terminal or (t==local_steps_per_epoch-1):
                if not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len)
                last_val = r if d else model.get_v(tf.constant(o.reshape(1, -1))).numpy()[0]
                buf.finish_path(last_val)
                if terminal:
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        # Perform PPO update!
        obs, acs, advs, rets, logp_olds = buf.get()

        obs = tf.constant(obs)
        acs = tf.constant(acs)
        advs = tf.constant(advs)
        rets = tf.constant(rets)
        logp_olds = tf.constant(logp_olds)
      
        pi_loss, v_loss, kl, ent, delta_pi_loss, delta_v_loss, stopIter = update(obs, acs, advs, rets, logp_olds)

        logger.store(LossPi=pi_loss.numpy(), LossV=v_loss.numpy(), KL=kl.numpy(), Entropy=ent.numpy(), DeltaLossPi=delta_pi_loss.numpy(), DeltaLossV=delta_v_loss.numpy(), StopIter=stopIter.numpy())

        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='ppo')
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    ppo(lambda : gym.make(args.env), actor_critic=core.mlp_actor_critic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma, 
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
        logger_kwargs=logger_kwargs)