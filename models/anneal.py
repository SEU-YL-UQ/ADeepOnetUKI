import jax.numpy as jnp 




class Annealing:
    """This class implements various annealing methods for the total training"""
    def __init__(self, total_epoch) -> None:
        self.total_epoch = total_epoch 
    
    def fast_annealing(self, current_epoch, last_epoch):
        cur_epoch = current_epoch - last_epoch 
        proportion = 1 - cur_epoch/(self.total_epoch - last_epoch)
        return proportion
    
    def cosine_annealing(self, current_epoch, last_epoch):
        cur_epoch = current_epoch - last_epoch 
        proportion = 0.5*(1 + jnp.sin(cur_epoch/(self.total_epoch - last_epoch) * jnp.pi)) 
        return proportion
    
    def exp_annealing(self, current_epoch, last_epoch, alpha = 0.1):
        proportion = jnp.exp(-alpha*(current_epoch - last_epoch))
        return proportion
    
    def boltzman_annealing(self, current_epoch, last_epoch, alpha = 0.9):
        proportion = alpha**(current_epoch - last_epoch)
        return proportion
    
        
    
    