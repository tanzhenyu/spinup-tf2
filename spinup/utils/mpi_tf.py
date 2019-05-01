import numpy as np
import tensorflow as tf
from mpi4py import MPI
from spinup.utils.mpi_tools import broadcast


def flat_concat(xs):
    return tf.concat([tf.reshape(x,(-1,)) for x in xs], axis=0)

def assign_params_from_flat(x, params):
    flat_size = lambda p : int(np.prod(p.shape.as_list())) # the 'int' is important for scalars
    splits = tf.split(x, [flat_size(p) for p in params])
    new_params = [tf.reshape(p_new, p.shape) for p, p_new in zip(params, splits)]
    return tf.group([tf.assign(p, p_new) for p, p_new in zip(params, new_params)])

def sync_params(params):
    get_params = flat_concat(params)
    def _broadcast(x):
        broadcast(x)
        return x
    synced_params = tf.py_function(_broadcast, [get_params], tf.float32)
    return assign_params_from_flat(synced_params, params)


class MpiAdamOptimizer(tf.optimizers.Adam):
    """
    Adam optimizer that averages gradients across MPI processes.

    The compute_gradients method is taken from Baselines `MpiAdamOptimizer`_. 
    For documentation on method arguments, see the Tensorflow docs page for 
    the base `AdamOptimizer`_.

    .. _`MpiAdamOptimizer`: https://github.com/openai/baselines/blob/master/baselines/common/mpi_adam_optimizer.py
    .. _`AdamOptimizer`: https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer
    """

    def __init__(self, **kwargs):
        self.comm = MPI.COMM_WORLD
        tf.optimizers.Adam.__init__(self, **kwargs)

    @tf.function
    def apply_gradients(self, grads_and_vars, name=None):
        """
        Same as normal apply_gradients, except sync params after update.
        """
        grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None]
        flat_grad = flat_concat([g for g, v in grads_and_vars])
        shapes = [v.shape.as_list() for g, v in grads_and_vars]
        sizes = [int(np.prod(s)) for s in shapes]

        num_tasks = self.comm.Get_size()
        buf = np.zeros(flat_grad.shape, np.float32)

        def _collect_grads(flat_grad):
            self.comm.Allreduce(flat_grad, buf, op=MPI.SUM)
            np.divide(buf, float(num_tasks), out=buf)
            return buf

        avg_flat_grad = tf.py_function(_collect_grads, [flat_grad], tf.float32)
        avg_flat_grad.set_shape(flat_grad.shape)
        avg_grads = tf.split(avg_flat_grad, sizes, axis=0)
        avg_grads_and_vars = [(tf.reshape(g, v.shape), v)
                    for g, (_, v) in zip(avg_grads, grads_and_vars)]
        opt = super().apply_gradients(avg_grads_and_vars, name)
        sync = sync_params([v for g,v in grads_and_vars])
        return tf.group([opt, sync])
