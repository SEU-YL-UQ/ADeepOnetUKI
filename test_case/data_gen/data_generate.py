from fenics import *
import numpy as np
import jax.numpy as jnp
from jax import random, config
from tqdm import tqdm 
from itertools import combinations
from scipy.interpolate import interpn
from typing import List



class RegularDomain:
    """This class is used to generate the samples in the boundary"""
    def __init__(self, space_domain, time_domain = None) -> None:
        """
        kwargs are the domain of every variable in the regular domain
        """
        self.space_domain = space_domain.reshape(-1, 2) 
        self.space_dim = self.space_domain.shape[0]
        self.bc_dim = 2*self.space_dim
        if time_domain is not None:
            self.bc_dim = 2**self.space_dim
        self.time_domain = time_domain
        self.space_mesh = jnp.meshgrid(*self.space_domain)
        self.space_points = jnp.array([mesh.flatten() for mesh in self.space_mesh]).T
        
    
    def generate_boundary_samples(self, num_samples, key, keep_same = False):
        vertex_num = num_samples//self.bc_dim
        if self.time_domain is not None:
            shape = (vertex_num, self.bc_dim)
            uniform_samples = self.uniform(key, shape, self.time_domain)
            if keep_same:
                uniform_samples = jnp.repeat(uniform_samples[:,0:1], axis = 1, repeats=self.bc_dim)
            Space_points = jnp.repeat(self.space_points, vertex_num, axis = 0)
            samples = jnp.hstack([Space_points, uniform_samples.reshape(-1,1)])
        else:
            if self.space_dim == 1:
                return self.space_points
            subkeys = random.split(key, self.bc_dim)
            samples = []
            for i in range(self.space_dim):
                select_domain = self.space_domain[i]
                domain = jnp.delete(self.space_domain, i, axis = 0)
                left = self.generate_interior_samples(domain, vertex_num, subkeys[i])
                right = self.generate_interior_samples(domain, vertex_num, subkeys[i + self.space_dim])
                if keep_same:
                    sample = jnp.vstack([left, left])
                else:
                    sample = jnp.vstack([left, right])
                space_point = jnp.repeat(select_domain, axis = 0, repeats=vertex_num)
                sample = jnp.insert(sample, i, space_point, axis = 1)
                samples.append(sample)
            samples = jnp.vstack(samples)
        return samples
    
    def generate_initial_samples(self, num_samples, key):
        if self.time_domain is None:
            return jnp.array([])
        samples = self.generate_interior_samples(self.space_domain, num_samples, key)
        samples = jnp.hstack([samples, jnp.zeros((num_samples, 1))])
        return samples
    
    def generate_residual_samples(self, num_samples, key):
        if self.time_domain is not None:
            domain = jnp.vstack([self.space_domain, self.time_domain])
            samples = self.generate_interior_samples(domain, num_samples, key)
        else:
            samples = self.generate_interior_samples(self.space_domain, num_samples, key)
        return samples
    
    def generate_interior_samples(self, domain, num_samples, key):
        domain = domain.reshape(-1, 2)
        dim = domain.shape[0]
        subkeys = random.split(key, dim)
        samples = []
        shape = (num_samples, 1)
        for i in range(dim):
            samples.append(self.uniform(subkeys[i], shape, domain[i]))
        return jnp.hstack(samples)
    
    def uniform(self, key, shape, domain = None):
        if domain is None:
            domain = [0, 1]
        return random.uniform(key, shape, minval = domain[0], maxval = domain[1])
        
    
    

class Data_generate:
    """This class implements data generating"""
    def __init__(self) -> None:
        """
            para:
               V: the functionspace for the prior
               prior: the prior distribution
               input_transform: the input_transform for the KL samples
               Output_tranform: the transform for the output values
        """
    
    def _toList(self, value):
        if not isinstance(value, List):
            value = [value - 1]
        else:
            value = [i - 1 for i in value]
        return value
    
    def flow_f(self, nodes):
        f = np.zeros((nodes.shape[0], 1))
        f[(nodes[:,1]>=0) & (nodes[:,1]<=2/3)] = 1000
        f[(nodes[:,1]>2/3) & (nodes[:,1]<=5/6)] = 2000
        f[(nodes[:,1]>5/6) & (nodes[:,1]<=1)] = 3000
        return jnp.asarray(f)
        
    
    def generate_PI_diffusion(self, prior,
                                randn_samples,
                                num_points,
                                space_domain, 
                                time_domain,
                                prior_discrete,
                                test = False,
                                update = False,
                                random_seed = 1234):
        
        """
        prior: the random field for generating the prior samples
        randn_samples: the randn samples for the random field
        num_points: the number of boundary, initial, residual points
        space_domain: the space discrete for prior
        time_domain: the time discrete of the pde
        space_discrete: the space discrete for the prior
        prior_discrete: the dimension for generating the prior samples
        test: whether to generate test samples
        update: whether to generate update samples
        """
        key = random.PRNGKey(random_seed)
        space_points = []
        num_samples = randn_samples.shape[0]
        space_domain = space_domain.reshape(-1, 2)
        dataset = RegularDomain(space_domain, time_domain)
        
        # space_discrete = self._toList(space_discrete)
        prior_discrete = self._toList(prior_discrete)
            
        
        for index in range(space_domain.shape[0]):
            points = np.linspace(*space_domain[index], prior_discrete[index] + 1)
            space_points.append(points)
            
        if dataset.space_dim == 1:
            # mesh = IntervalMesh(*space_discrete, *space_domain[0])
            prior_mesh = IntervalMesh(*prior_discrete, *space_domain[0])
        else:
            point_left = Point(space_domain[:, 0])
            point_right = Point(space_domain[:, 1])
            # mesh = RectangleMesh(point_left, point_right, *space_discrete)  
            prior_mesh = RectangleMesh(point_left, point_right, *prior_discrete)  
        prior_V = FunctionSpace(prior_mesh, 'P', 1)
        
        total_num = num_points[0] + num_points[1]
        num_bc, num_initial, num_res = num_points
        
        U_res_train = jnp.zeros((num_samples * num_res, randn_samples.shape[1]))
        Y_res_train = jnp.zeros((num_samples * num_res, dataset.space_dim + 1))
        S_res_train = jnp.zeros((num_samples * num_res, 1))
        
        U_train = jnp.zeros((num_samples * total_num, randn_samples.shape[1]))
        Y_train = jnp.zeros((num_samples * total_num, dataset.space_dim + 1)) 
        S_train = jnp.zeros((num_samples * total_num, 1))
        
        U_test = []
        prior_samples = prior.sample(prior_V, randn_samples)
            
        subkeys = random.split(key, num_samples)
        
        
        for i in tqdm(range(num_samples)):
            u_fn = lambda x: interpn(space_points, prior_samples[i], x, method = 'cubic')
            # prior_train = u_fn(select_points.reshape(-1, dataset.space_dim)).reshape(1,-1)
            boundary_points = dataset.generate_boundary_samples(num_bc, subkeys[i])
            initial_points = dataset.generate_initial_samples(num_initial, subkeys[i])
            residual_points = dataset.generate_residual_samples(num_res, subkeys[i])
            
            u_train = jnp.tile(randn_samples[i], (total_num, 1))
            u_res = jnp.tile(randn_samples[i], (num_res, 1))
            s_res = u_fn(residual_points[:,0:dataset.space_dim]).reshape(-1,1)
            s_train = jnp.zeros((total_num, 1))
            y_train = jnp.vstack([boundary_points, initial_points])
            
            U_test.append(randn_samples[i])
            U_train = U_train.at[i*total_num:(i+1)*total_num].set(u_train)
            Y_train = Y_train.at[i*total_num:(i+1)*total_num].set(y_train) 
            S_train = S_train.at[i*total_num:(i+1)*total_num].set(s_train)
            U_res_train = U_res_train.at[i*num_res:(i+1)*num_res].set(u_res)
            Y_res_train = Y_res_train.at[i*num_res:(i+1)*num_res].set(residual_points) 
            S_res_train = S_res_train.at[i*num_res:(i+1)*num_res].set(s_res)
        
        
            
        if test:
            return jnp.vstack(U_test)
        elif update:
            return U_res_train, Y_res_train, S_res_train
        else:
            return U_train, Y_train, S_train, U_res_train, Y_res_train, S_res_train
    
    def generate_PI_flow(self, prior,
                            randn_samples,
                            num_points,
                            space_domain, 
                            prior_discrete,
                            test = False,
                            update = False,
                            random_seed = 1234):
        
        """
        prior: the random field for generating the prior samples
        randn_samples: the randn samples for the random field
        num_points: the number of boundary, initial, residual points
        space_domain: the space discrete for prior
        time_domain: the time discrete of the pde
        space_discrete: the space discrete for the prior
        prior_discrete: the dimension for generating the prior samples
        test: whether to generate test samples
        update: whether to generate update samples
        """
        key = random.PRNGKey(random_seed)
        space_points = []
        num_samples = randn_samples.shape[0]
        space_domain = space_domain.reshape(-1, 2)
        dataset = RegularDomain(space_domain)
        
        # space_discrete = self._toList(space_discrete)
        prior_discrete = self._toList(prior_discrete)
            
        
        for index in range(space_domain.shape[0]):
            points = np.linspace(*space_domain[index], prior_discrete[index] + 1)
            space_points.append(points)
            
        if dataset.space_dim == 1:
            # mesh = IntervalMesh(*space_discrete, *space_domain[0])
            prior_mesh = IntervalMesh(*prior_discrete, *space_domain[0])
        else:
            point_left = Point(space_domain[:, 0])
            point_right = Point(space_domain[:, 1])
            # mesh = RectangleMesh(point_left, point_right, *space_discrete)  
            prior_mesh = RectangleMesh(point_left, point_right, *prior_discrete)  
        prior_V = FunctionSpace(prior_mesh, 'P', 1)
        
        total_num = num_points[0] + num_points[1]
        num_bc, num_initial, num_res = num_points
        
        U_res_train = jnp.zeros((num_samples * num_res, randn_samples.shape[1]))
        Y_res_train = jnp.zeros((num_samples * num_res, dataset.space_dim + 3))
        S_res_train = jnp.zeros((num_samples * num_res, 1))
        
        U_train = jnp.zeros((num_samples*total_num, randn_samples.shape[1]))
        Y_train = jnp.zeros((num_samples * total_num, dataset.space_dim)) 
        S_train = jnp.zeros((num_samples * total_num, 1))
        
        U_test = []
        prior_samples = prior.sample(prior_V, randn_samples)
        prior_samples_x = prior.sample(prior_V, randn_samples, derivative = 1)
        prior_samples_y = prior.sample(prior_V, randn_samples, derivative = 2)
            
        subkeys = random.split(key, num_samples)
        
        
        for i in tqdm(range(num_samples)):
            u_fn = lambda x: interpn(space_points, prior_samples[i], x, method = 'cubic')
            u_fn_x = lambda x: interpn(space_points, prior_samples_x[i], x, method = 'cubic')
            u_fn_y = lambda x: interpn(space_points, prior_samples_y[i], x, method = 'cubic')
            # prior_train = u_fn(select_points.reshape(-1, dataset.space_dim)).reshape(1,-1)
            boundary_points = dataset.generate_boundary_samples(num_bc, subkeys[i])
            residual_points = dataset.generate_residual_samples(num_res, subkeys[i])
            
            u_train = jnp.tile(randn_samples[i], (total_num, 1))
            u_res = jnp.tile(randn_samples[i], (num_res, 1))
            s_res = self.flow_f(residual_points)
            s_train = jnp.zeros((total_num, 1))
            y_train = boundary_points
            kappa = u_fn(residual_points).reshape(-1,1)
            kappa_x = u_fn_x(residual_points).reshape(-1,1)
            kappa_y = u_fn_y(residual_points).reshape(-1,1)
            y_res = jnp.hstack([residual_points, kappa, kappa_x, kappa_y])
            
            U_test.append(randn_samples[i])
            U_train = U_train.at[i*total_num:(i+1)*total_num].set(u_train)
            Y_train = Y_train.at[i*total_num:(i+1)*total_num].set(y_train) 
            S_train = S_train.at[i*total_num:(i+1)*total_num].set(s_train)
            U_res_train = U_res_train.at[i*num_res:(i+1)*num_res].set(u_res)
            Y_res_train = Y_res_train.at[i*num_res:(i+1)*num_res].set(y_res) 
            S_res_train = S_res_train.at[i*num_res:(i+1)*num_res].set(s_res)
                
        if test:
            return jnp.vstack(U_test)
        elif update:
            return U_res_train, Y_res_train, S_res_train
        else:
            return U_train, Y_train, S_train, U_res_train, Y_res_train, S_res_train
    
    def generate_deeponet(self, prior, forward_sover, randn_samples, prior_V,
                 num_sensors, random_seed = 1234):
        
        key = random.PRNGKey(random_seed)
        num_samples = randn_samples.shape[0]
        space_dim = prior_V.mesh().coordinates().shape[1]
        prior_mesh = prior_V.mesh()
        
        
        prior_samples = prior.sample(prior_V, randn_samples)
        subkeys = random.split(key, num_samples)
        dim = prior_V.dim()
        
        num_train = num_samples 
        U_train = jnp.zeros((num_train * num_sensors, randn_samples.shape[1]))
        Y_train = jnp.zeros((num_train * num_sensors, space_dim)) 
        S_train = jnp.zeros((num_train * num_sensors, 1))
        S_true = []
        
        for i in tqdm(range(num_train)):
            solution = forward_sover(prior_samples[i])
            S_true.append(solution)
            # prior_sample = prior_samples[i][::step, ::step]
            u_train =  jnp.tile(randn_samples[i].reshape(1, -1), (num_sensors, 1))
            select_index = random.choice(subkeys[i], jnp.arange(dim), (num_sensors,), False)
            sensors = prior_mesh.coordinates()[select_index]
            s_train = solution[select_index]
            U_train = U_train.at[i*num_sensors:(i+1)*num_sensors].set(u_train)
            Y_train = Y_train.at[i*num_sensors:(i+1)*num_sensors].set(sensors.squeeze()) 
            S_train = S_train.at[i*num_sensors:(i+1)*num_sensors].set(s_train.reshape(-1,1))
        return U_train, Y_train, S_train
    
    def generate_source(self, forward_solver, V, randn_samples, num_sensors, random_seed = 1234):
        key = random.PRNGKey(random_seed)
        num_samples = randn_samples.shape[0]
        num_train = num_samples 
        dim = V.dim()
        prior_mesh = V.mesh()
        U_train = jnp.zeros((num_train * num_sensors, randn_samples.shape[1]))
        Y_train = jnp.zeros((num_train * num_sensors, 2)) 
        S_train = jnp.zeros((num_train * num_sensors, 1))
        U_train1 = jnp.zeros((num_train * num_sensors, randn_samples.shape[1]))
        S_train1 = jnp.zeros((num_train * num_sensors, 1))
        subkeys = random.split(key, num_samples)
        for i in tqdm(range(num_train)):
            solution = forward_solver(randn_samples[i])
            # prior_sample = prior_samples[i][::step, ::step]
            u_train =  jnp.tile(randn_samples[i].reshape(1, -1), (num_sensors, 1))
            select_index = random.choice(subkeys[i], jnp.arange(dim), (num_sensors,), False)
            sensors = prior_mesh.coordinates()[select_index]
            s_train = solution[0][select_index]
            s_train1 = solution[1][select_index]
            U_train = U_train.at[i*num_sensors:(i+1)*num_sensors].set(u_train)
            U_train1 = U_train1.at[i*num_sensors:(i+1)*num_sensors].set(u_train)
            Y_train = Y_train.at[i*num_sensors:(i+1)*num_sensors].set(sensors.squeeze()) 
            S_train = S_train.at[i*num_sensors:(i+1)*num_sensors].set(s_train.reshape(-1,1))
            S_train1 = S_train1.at[i*num_sensors:(i+1)*num_sensors].set(s_train1.reshape(-1,1))
        return U_train, Y_train, S_train, U_train1, Y_train, S_train1
        
        
        
        
    
            
                
            
            
            
            
        
        
        
        
    
    
    
        
            
            
    
    


    
    
            
            
            
        
            
      
      
      
      



      
      
        