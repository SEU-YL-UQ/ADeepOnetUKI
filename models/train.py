import jax.numpy as np
import jax
from jax import random, grad, vmap, jit 
from jax.flatten_util import ravel_pytree
from jax.example_libraries import optimizers
import itertools
import os 
import pickle 
import matplotlib.pyplot as plt 
from functools import partial 
from tqdm import trange

from .network import MLP, Linear_MLP
from utils.earlystopping import EarlyStopping


class Train:
    """
    This is the mother class for the following training method
    """
    def __init__(self, branch_layers, trunk_layers, save_path, linear = False) -> None:
        self.branch_layers = branch_layers
        self.trunk_layers = trunk_layers
        self.save_path = save_path 
        if linear:
            self.branch_init, self.branch_apply = Linear_MLP(branch_layers)
            self.trunk_init, self.trunk_apply = Linear_MLP(trunk_layers)
        else:
            self.branch_init, self.branch_apply = MLP(branch_layers, activation=jax.nn.tanh)  # or Relu 
            self.trunk_init, self.trunk_apply = MLP(trunk_layers, activation=jax.nn.tanh)     # or Relu
        self.dim = trunk_layers[0]

        # Initialize
        branch_params = self.branch_init(rng_key = random.PRNGKey(1234))
        trunk_params = self.trunk_init(rng_key = random.PRNGKey(4321))
        params = (branch_params, trunk_params)

        # Use optimizers to set optimizer initialization and update functions
        self.opt_init, \
        self.opt_update, \
        self.get_params = optimizers.adam(optimizers.exponential_decay(1e-3, 
                                                                      decay_steps=2000, 
                                                                      decay_rate=0.95))
        self.opt_state = self.opt_init(params)

        # Used to restore the trained model parameters
        _, self.unravel_params = ravel_pytree(params)

        self.itercount = itertools.count()
        self.iter = 0

        # Loggers
        self.loss_log = []
        self.early_stopping = EarlyStopping(10)

        
    
    def operator_net(self, params, u, *sensors):
        branch_params, trunk_params = params
        y = np.stack(sensors)
        B = self.branch_apply(branch_params, u)
        T = self.trunk_apply(trunk_params, y)
        outputs = np.sum(B * T)
        return  outputs
    
    
    
    @partial(jit, static_argnums=(0,))
    def predict_s(self, params, U_star, Y_star):
        
        s_pred = vmap(self.operator_net, (None,0) + (0, )*self.dim)(params, U_star, *Y_star.T)
        return s_pred
    
    
    def visual_loss(self, loss, prefix):
        plt.style.use('default')
        plt.yscale('log')
        plt.tight_layout()
        for name, value in loss.items():
            if name != 'epoch' and len(value)>0:
                plt.plot(loss['epoch'], value, label = name)
        
        plt.title('Epoch:' + str(prefix))
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(self.save_path.img_save_path, str(prefix) + '.png'))
        plt.close()
    
    
    def save_data(self, params, save_path, prefix):
        save_path = os.path.join(save_path, str(prefix) + '.npy')
        with open(save_path, 'wb') as f:
            pickle.dump(params, f)
    
    def load_data(self, save_path, prefix):
        save_path = os.path.join(save_path, prefix + '.npy')
        with open(save_path, 'rb') as f:
            return pickle.load(f)
    
    def save_model(self, prefix):
        self.save_data(self.get_params(self.opt_state), self.save_path.model_save_path, prefix)
    
    def load_model(self, prefix):
        if os.path.exists(os.path.join(self.save_path.model_save_path, prefix + '.npy')):
            with open(os.path.join(self.save_path.model_save_path, prefix + '.npy'), 'rb') as f:
                params = pickle.load(f)
        
        self.opt_state = self.opt_init(params)
        if prefix == 'checkpoint':
            self.base_params = params 
            
        
            
             
        
        




class DeepONet(Train):
    def __init__(self, branch_layers, trunk_layers, save_path, linear = False):  
        super().__init__(branch_layers, trunk_layers, save_path, linear)  
        # Network initialization and evaluation functions
        self.loss_log = {"loss": [], 'epoch':[]}

    # Define DeepONet architecture
    
  
    # Define operator loss
    def loss_operator(self, params, batch):
        inputs, outputs = batch
        u, y = inputs
        # Compute forward pass
        s_pred = self.predict_s(params, u, y)
        # Compute loss
        loss = np.mean((outputs.flatten() - s_pred.flatten())**2)
        return loss

    # Define a compiled update step
    @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state, batch):
        params = self.get_params(opt_state)
        g = grad(self.loss_operator)(params, batch)
        return self.opt_update(i, g, opt_state)

    # Optimize parameters in a loop
    def train(self, dataset, nIter = 10000, update = False, decay_steps = None):
        # Define data iterators
        if decay_steps is None: decay_steps = 500

        data_iterator = iter(dataset)
        if update:
            self.itercount = itertools.count()
            self.opt_init, \
            self.opt_update, \
            self.get_params = optimizers.adam(optimizers.exponential_decay(5e-4, 
                                                                        decay_steps = 500, 
                                                                        decay_rate=0.9))
        # testdata_iterator = iter(test_dataset)
        pbar = trange(nIter)
        # Main training loop
        for it in pbar:
            # Fetch data
            batch = next(data_iterator)
            self.opt_state = self.step(next(self.itercount), self.opt_state, batch)
            
            if it % 1000 == 0:
                params = self.get_params(self.opt_state)

                # Compute loss
                loss_value = self.loss_operator(params, batch)
                
                # Store loss
                self.loss_log['loss'].append(loss_value)
                
                self.loss_log['epoch'].append(it + self.iter)
                self.visual_loss(self.loss_log, 'loss')
                pbar.set_postfix({'Loss': loss_value})
                
        self.iter += nIter
    
        
        
           
    # Evaluates predictions at test points  
class PI_DeepONet(Train):
    def __init__(self, branch_layers, trunk_layers, save_path): 
        super().__init__(branch_layers, trunk_layers, save_path)   
        # Network initialization and evaluation functions
        self.loss_log = {"loss": [], "bc_loss": [], "res_loss": [], 'epoch':[]}
        
    # Define DeepONet architecture
  
    # Define ODE/PDE residual
    def residual_net(self, params, u, *sensors):
        pass 
        

    # Define boundary loss
    def loss_bcs(self, params, batch):
        inputs, outputs = batch
        u, y = inputs

        # Compute forward pass
        s_pred = self.predict_s(params, u, y)
        # s_pred = vmap(self.operator_net, (None, 0, 0, 0))(params, u, y[:,0], y[:,1])

        # Compute loss
        loss = np.mean((outputs.flatten() - s_pred)**2)
        return loss

    # Define residual loss
    def loss_res(self, params, batch):
        # Fetch data
        # inputs: (u1, y), shape = (Nxm, m), (Nxm,1)
        # outputs: u2, shape = (Nxm, 1)
        inputs, outputs = batch
        u, y = inputs
        # Compute forward pass
        # pred = vmap(self.residual_net, (None, 0, 0, 0))(params, u, y[:,0], y[:,1])
        pred = self.predict_res(params, u, y)

        # Compute loss
        loss = np.mean((outputs.flatten() - pred)**2)
        return loss   

    # Define total loss
    def loss(self, params, bcs_batch, res_batch):
        loss_bcs = self.loss_bcs(params, bcs_batch)
        loss_res = self.loss_res(params, res_batch)
        loss = loss_bcs + 0.001*loss_res
        return loss 

    # Define a compiled update step
    @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state, bcs_batch, res_batch):
        params = self.get_params(opt_state)
        g = grad(self.loss)(params, bcs_batch, res_batch)
        return self.opt_update(i, g, opt_state)

    # Optimize parameters in a loop
    def train(self, bcs_dataset, res_dataset, nIter = 10000, update = False):
        # Define data iterators
        bcs_data = iter(bcs_dataset)
        res_data = iter(res_dataset)
        if update: 
            self.itercount = itertools.count()
        pbar = trange(nIter)
        # Main training loop
        for it in pbar:
            # Fetch data
            bcs_batch= next(bcs_data)
            res_batch = next(res_data)

            self.opt_state = self.step(next(self.itercount), self.opt_state, bcs_batch, res_batch)
            
            if it % 100 == 0 or it == 0:
                params = self.get_params(self.opt_state)

                # Compute losses
                loss_value = self.loss(params, bcs_batch, res_batch)
                loss_bcs_value = self.loss_bcs(params, bcs_batch)
                loss_res_value = 0.01*self.loss_res(params, res_batch)

                # Store losses
                self.loss_log['loss'].append(loss_value)
                self.loss_log['bc_loss'].append(loss_bcs_value)
                self.loss_log['res_loss'].append(loss_res_value)
                self.loss_log['epoch'].append(it + self.iter)

                # Print losses
                pbar.set_postfix({'Loss': loss_value, 
                                  'loss_bcs' : loss_bcs_value, 
                                  'loss_res': loss_res_value})
                
                self.visual_loss(self.loss_log, 'loss')
                
                
        self.save_data(self.loss_log, 'loss')
                    
        
        self.iter += nIter
    
    
           
    # Evaluates predictions at test points  
    @partial(jit, static_argnums=(0,))
    def predict_res(self, params, U_star, Y_star):
        r_pred = vmap(self.residual_net, (None,0) + (0, )*Y_star.shape[1])(params, U_star, *Y_star.T)
        return r_pred

class DNN_train:
    """This class implements the dnn training procedure using jax"""
    def __init__(self, layers, save_path) -> None:
        self.layers = layers 
        self.save_path = save_path 
        self.dnn_init, self.dnn_apply = Linear_MLP(layers)
        params = self.dnn_init(rng_key = random.PRNGKey(1234))
        self.opt_init, \
        self.opt_update, \
        self.get_params = optimizers.adam(optimizers.exponential_decay(1e-3, 
                                                                      decay_steps=400, 
                                                                      decay_rate=0.9))
        self.itercount = itertools.count()
        self.opt_state = self.opt_init(params)
        self.loss_log = {'loss':[], 'epoch':[]}
    
    def loss(self, params, batch):
        y, u = batch 
        pred = self.dnn_apply(params, y)
        return np.mean((u.reshape(pred.shape) - pred)**2)
    
    @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state, batch):
        params = self.get_params(opt_state)
        g = grad(self.loss)(params, batch)
        return self.opt_update(i, g, opt_state)
    
    def train(self, dataset, nIter = 10000, prefix = None):
        data = iter(dataset)
        pbar = trange(nIter)
        for i in pbar:
            batch = next(data)
            self.opt_state = self.step(next(self.itercount), self.opt_state, batch)
            if i % 100 == 0:
                params = self.get_params(self.opt_state)
                loss_value = self.loss(params, batch)
                self.loss_log['loss'].append(loss_value)
                self.loss_log['epoch'].append(i)
                pbar.set_postfix({'Loss': loss_value})
                self.visual_loss(self.loss_log, 'loss_dnn')
        self.save_data(self.get_params(self.opt_state), self.save_path.model_save_path, prefix)
    
    def visual_loss(self, loss, prefix):
        plt.style.use('default')
        plt.figure()
        plt.yscale('log')
        plt.tight_layout()
        for name, value in loss.items():
            if name != 'epoch' and len(value)>0:
                plt.plot(loss['epoch'], value, label = name)
        
        plt.title('Epoch:' + str(prefix))
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(self.save_path.img_save_path, str(prefix) + '.png'))
        plt.close()
    
    def save_data(self, params, save_path, prefix):
        save_path = os.path.join(save_path, str(prefix) + '.npy')
        with open(save_path, 'wb') as f:
            pickle.dump(params, f)
    
    def load_model(self, prefix):
        if os.path.exists(os.path.join(self.save_path.model_save_path, prefix + '.npy')):
            with open(os.path.join(self.save_path.model_save_path, prefix + '.npy'), 'rb') as f:
                params = pickle.load(f)
        self.opt_state = self.opt_init(params)
        
            
    @partial(jit, static_argnums = (0, ))
    def predict(self, params, y):
        return self.dnn_apply(params, y)
    
    
        
            
            
            
            
    


                