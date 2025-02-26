import numpy as np
import jax.numpy as jnp 
from jax import random, config, vmap, jit
from sklearn.gaussian_process.kernels import RBF
from fenics import * 
import matplotlib.pyplot as plt




class GaussianRF_KL:
    """This class generates the gaussian random field samples using KL expansion"""
    def __init__(self, tau, alpha, sigma = None) -> None:
        self.tau = tau 
        self.alpha = alpha 
        self.sigma = sigma 
        self.type = 'KL'
        
    
    def sample(self, V, rand_samples, space_mesh = None, derivative = False):
        """
        parameter:
            space_dim: the dimension of the space dicretion
            num_kl: the number for KL terms
            space_mesh: the space points we want to compute
            derivative: whether we want to compute the derivative through the kl expansion, the value should be 1, 2
        """
        self.space_dim = V.mesh().geometric_dimension()
        if len(rand_samples.shape) == 1:
            num_kl = len(rand_samples)
        else:
            num_kl = rand_samples.shape[1]
        rand_samples = rand_samples.reshape(-1, num_kl)
            
        if self.space_dim == 1:
            shape = (V.dim(), 1)
        else:
            shape = jnp.int32(jnp.array([jnp.sqrt(V.dim())]*2))
        if self.sigma is None:
            self.sigma = self.tau**(0.5*(2*self.alpha-self.space_dim))
            
        space_points = V.mesh().coordinates()
        if space_mesh is None:
            space_mesh = [space_points[:,i].reshape(shape) for i in range(self.space_dim)]
        
        sqrt_eigenvals, kl_points = self.eigen_val(num_kl)
        ###select the corresponding eigenvalues
        index = jnp.argsort(sqrt_eigenvals)[::-1][:num_kl]
        self.select_eigenvals = sqrt_eigenvals[index]
        # print(select_eigenvals)
        select_kl_points = kl_points[index]
        ###generate the corresponding eigen functions
        self.eigen_funcs = self.eigen_func(select_kl_points, space_mesh, derivative)
        # np.save('./eigen_func', eigen_funcs)
        
        f = lambda sample: jnp.einsum('j,j,jmk->mk', sample, self.select_eigenvals, self.eigen_funcs)
        
        sample = vmap(f, (0))(rand_samples)
        return sample
    
    def eigen_func(self, kl_point, mesh, derivative = False):
        phi = jnp.zeros((len(kl_point), *mesh[0].shape))
        for i in range(len(kl_point)):
            if kl_point[i,0] == 0 and kl_point[i,1] == 0:
                phi = phi.at[i].set(1)
            elif kl_point[i,0] == 0:
                phi = phi.at[i].set(jnp.sqrt(2)*jnp.cos(jnp.pi * (kl_point[i, 1]*mesh[1])))
            elif kl_point[i,1] == 0:
                phi = phi.at[i].set(jnp.sqrt(2)*jnp.cos(jnp.pi * (kl_point[i, 0]*mesh[0])))
            else:
                phi = phi.at[i].set(2*jnp.cos(jnp.pi * (kl_point[i, 0]*mesh[0])) *  jnp.cos(jnp.pi * (kl_point[i, 1]*mesh[1])))
        return phi
    
    def eigen_val(self, num_kl):
        if self.space_dim == 2:
            kl_dim = jnp.int32(jnp.sqrt(2*num_kl)) + 1
            kl_mesh = jnp.meshgrid(*[jnp.arange(kl_dim)]*self.space_dim)
            kl_points = jnp.array([mesh.flatten() for mesh in kl_mesh]).T
            kl_points = kl_points[1:]
        elif self.space_dim == 1:
            kl_points = jnp.arange(1, num_kl + 1)[:,None]
        
       
        kl_norm = jnp.linalg.norm(kl_points, axis = 1)**2
        # print(kl_norm)
        sqrt_eigenvals = self.sigma*(jnp.pi**2*kl_norm + self.tau**2)**(-self.alpha/2)
        return sqrt_eigenvals, kl_points


class GaussianRF_RBF:
    """This class create gaussian random field with simpler kernels"""
    def __init__(self, scale) -> None:
        self.scale = scale 
        self.kernel = RBF(scale)
        self.type = "RBF"
    
    def sample(self, V, randn_samples):
        "This function helps generate samples with zero mean"
        config.update('jax_enable_x64', True)
        nodes = V.mesh().coordinates()
        kernel_matrix = self.kernel(nodes, nodes) + 1e-10*jnp.eye(V.dim())
        R = jnp.linalg.cholesky(kernel_matrix)
        samples = randn_samples@R.T
        config.update('jax_enable_x64', False)
        return samples

# key = random.PRNGKey(123)
# rand_samples = random.normal(key, (10, 100))
# mesh = UnitIntervalMesh(60)
# V = FunctionSpace(mesh, 'P', 1)
# GRF = GaussianRF_KL(5, 4)
# sample = GRF.sample(V,100, rand_samples)
# plt.imshow(sample[0])
# plt.colorbar()
# plt.plot(sample[0])
# plt.savefig('./sample.png')
# print(np.sum(sample))











        



        
        

        

        

