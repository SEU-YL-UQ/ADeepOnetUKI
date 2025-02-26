import numpy as np
import os 
import jax.numpy as jnp
import shutil
import torch.utils.data as data_0
from torch.utils import data


class Data(object):
    """
    This class generate the data class for the training of deeponet
    """
    def __init__(
        self, U_train, Y_train, S_train, U_res, Y_res, S_res):
        """
        para:
           U_train: the input data for deeponet
           X_train: the sensors
           S_train: the output data for the deepont
           U_val: the test data
           X_val: the test sensor
           S_val: the test output, basically the solutions of the system
        """
        self.u = U_train
        self.y = Y_train
        self.s = S_train 
        self.s_res = S_res 
        self.y_res = Y_res 
        self.u_res = U_res
        
        
    @property
    def bc_dataset(self):
        return self.u, self.y, self.s 
    
    @property
    def res_dataset(self):
        return self.u_res, self.y_res, self.s_res
    
    def update(self, u_res, y_res, s_res, num_total, num_res, num_update):
        """This function prepares for addtional training for the deeponet"""
        old_num = num_total - num_update
        choice = np.random.choice(num_res, old_num, replace = False)
        choice = np.repeat(choice, num_res) + np.tile(np.arange(num_res), old_num)
        choice = jnp.asarray(choice)
        old_u = self.u_res[choice]
        old_y = self.y_res[choice]
        old_s = self.s_res[choice]
        self.u_res = jnp.vstack([old_u, u_res])
        self.y_res = jnp.vstack([old_y, y_res])
        self.s_res = jnp.vstack([old_s, s_res])
        
         
    
    
        

    
            
    

    


def delete(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.makedirs(path)


    
    
    


