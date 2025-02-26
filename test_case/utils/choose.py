import jax.numpy as jnp 
from jax import vmap
from tqdm import trange 
import numpy as np 
from jax import jit 

@jit
def point_to_set_max(point, set):
    return jnp.max(jnp.linalg.norm(point - set, axis = 1))

@jit
def point_to_set_min(point, set):
    return jnp.min(jnp.linalg.norm(point - set, axis = 1))


def choose_high(cur_mean, candidate, forward_model, num_points):
    cur_mean = cur_mean.reshape(1, -1)
    pbar = trange(num_points)
    sol = forward_model(candidate)
    d_cur = jnp.linalg.norm(candidate - cur_mean, axis = 1)
    # d_pre = jnp.linalg.norm(candidate - pre_mean, axis = 1)
    for i in pbar:
        if i == 0:
            index = jnp.argmax(-d_cur)
            sample_sol = sol[index].reshape(1, -1)
            samples = candidate[index].reshape(1,-1)
        else:
            d_sol = vmap(point_to_set_max, in_axes=(0, None))(sol, sample_sol)
            # d_sol = d_sol*d_cur.max()/d_sol.max()
            index = jnp.argmax(-d_cur + d_sol)
            sample_sol = jnp.vstack([sample_sol, sol[index]])
            samples = jnp.vstack([samples, candidate[index]])

        
        candidate = jnp.delete(candidate, index, axis = 0)
        sol = jnp.delete(sol, index, axis = 0)
        d_cur = jnp.delete(d_cur, index)
        # dis = vmap(point_to_set_min, in_axes=(0, None))(candidate, samples)
        # index = jnp.logical_not(dis > sigma)
        # candidate = jnp.delete(candidate, index, axis = 0)
        # sol = jnp.delete(sol, index, axis = 0)
        # d_cur = jnp.delete(d_cur, index, axis = 0)
        
        # candidate = candidate[index]
        # sol = sol[index]
        # d_cur = d_cur[index]
        # d_pre = d_pre[jnp.r_[0:index, index+1:length]]
        
    return samples



