import jax.numpy as jnp


def compute_mean_cov(predict_s, true_s, obs_dim):
    """
    model: the deeponet model
    dataset: the dataset for training
    """
    predict_s = predict_s.reshape(-1, obs_dim)
    true_s = true_s.reshape(-1, obs_dim)
    sample_mean = jnp.mean(true_s - predict_s, axis = 0)
    sample_cov = jnp.cov(true_s - predict_s, rowvar = False)
    return sample_mean, sample_cov
    
    
    
    
    