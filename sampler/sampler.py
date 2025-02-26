import numpy as np
import torch
import jax.numpy as jnp
import time 
from jax import config
import torch.autograd as autograd
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
from utils.plotters import Plotters


# torch.set_default_dtype(torch.float64)

class RBF(torch.nn.Module):
    def __init__(self, sigma=None):
        super(RBF, self).__init__()

        self.sigma = sigma

    def forward(self, X, Y):
        # A @ B
        # X,Y .shape [10, 2]
        if not isinstance(X, torch.Tensor) or not isinstance(Y, torch.Tensor):
            X = torch.from_numpy(X).float()
            Y = torch.from_numpy(Y).float()
        XX = X.matmul(X.t())
        XY = X.matmul(Y.t())
        YY = Y.matmul(Y.t())

        dnorm2 = -2 * XY + XX.diag().unsqueeze(1) + YY.diag().unsqueeze(0)  # 10x10

        # Apply the median heuristic (PyTorch does not give true median)
        if self.sigma is None:
            np_dnorm2 = dnorm2.detach().cpu().numpy()
            h = np.median(np_dnorm2) / (2 * np.log(X.size(0) + 1))
            sigma = np.sqrt(h).item()
        else:
            sigma = self.sigma

        gamma = 1.0 / (1e-8 + 2 * sigma ** 2)
        K_XY = (-gamma * dnorm2).exp()
        return K_XY

class SVGD:
    def __init__(self, log_prob_derivative, dim, device):
        self.K = RBF(sigma = 0.2)
        self.dim = dim
        self.log_prob_derivative = log_prob_derivative
        self.device = device

    def phi(self, X):
        X = X.detach().requires_grad_(True)

        log_prob_derivative = self.log_prob_derivative(X)
          # dlog(Px)/dx   [10, 2]

        K_XX = self.K(X, X.detach())   # [10, 10]
        grad_K = -autograd.grad(K_XX.sum(), X)[0]  # dPhi(x)/dx [10, 2]
        
		#  stein's indentity ([10, 10]x[10,2]+[10, 2])/10 ->[10, 2]
        phi = (K_XX.detach().matmul(log_prob_derivative) + grad_K) / X.size(0) 

        return phi

    def step(self, X):
        self.optim.zero_grad()
        X.grad = -self.phi(X)
        self.optim.step()
    
    def sample(self, num_samples, num_step = 1000):
        init_x = torch.rand(num_samples, self.dim).to(self.device)
        self.optim = torch.optim.Adam([init_x], lr = 0.01)
        print('Start svgd sampling--------------------------------------')
        for i in tqdm(range(num_step)):
            self.step(init_x)
        return init_x.to('cpu').detach()
    

class LangevinDynamics:
    """This class implements langevin sampling for the bayesian posterior distribution"""
    def __init__(self, log_prob_derivative, dim, device) -> None:
        self.log_prob_derivative = log_prob_derivative
        self.dim = dim 
        self.device = device
    
    def step(self, x):
        x = x - self.lr*self.log_prob_derivative(x) \
            + torch.sqrt(2*self.lr)*torch.randn(1, len(x)).to(self.device)
        return x 
    
    def sample(self, lr, num_step):
        init_x = torch.randn(1, self.dim).to(self.device)
        self.lr = torch.tensor(lr).to(self.device)
        samples = torch.empty((0, self.dim)).to(self.device)
        print('Start langevin sampling process')
        for i in tqdm(range(num_step)):
            init_x = self.step(init_x)
            samples = torch.vstack([samples, init_x])
        return samples.to('cpu').detach()
        
        

class UKI1:
    """This class implements uki"""
    def __init__(self, para_dim, obs_dim, gamma, sigma, alpha, delta_t = 0.5) -> None:
        """
            Para:
                para_dim: the dimension of the parameter space
                obs_dim: the dimensionl of the observation spave
                gamma: the parameter for the initial covariance
                alpha: the regularization parameter
                
        """
        self.para_dim = para_dim
        self.obs_dim = obs_dim
        self.alpha = alpha 
        self.delta_t = delta_t
        self.sigma = sigma
        self.init_cov = jnp.eye(para_dim)*gamma**2
      
        
        
         

    def get_sigma_points(self, mean, cov):
        """This file constructs the sigma points"""
        R = sqrtm(cov)
        points = jnp.zeros((self.num_sigma, self.para_dim))
        if len(self.c_weights.shape) == 1:
            points = points.at[0].set(mean)
            temp = jnp.einsum('i, ij->ij', self.c_weights, R.T)
            points = points.at[1:self.para_dim + 1].set(mean + temp)
            points = points.at[self.para_dim + 1:].set(mean - temp)
        elif len(self.c_weights.shape) == 2:
            points = points.at[0].set(mean) 
            points = points.at[1:].set(mean + jnp.dot(R, self.c_weights))
        return points 

    def prediction(self, mean, cov):
        """This function constructs the prediction step"""
        mean_hat = self.alpha * mean + (1 - self.alpha) * self.init_mean
        cov_hat = self.alpha**2 * cov + self.sigma_predict
        return mean_hat, cov_hat 

    def analysis(self, forward, mean, cov, obs):
        ###prediction step
        mean_hat, cov_hat = self.prediction(mean, cov)
        ###construct emsemble
        sigma_points = self.get_sigma_points(mean_hat, cov_hat)
        y_hat = forward(sigma_points)
        y_hat_mean = jnp.einsum('j, ji->i', self.mean_weights, y_hat)
        ###construct error
        error_y = y_hat - y_hat_mean
        cov_theta_y = jnp.einsum('b, bi, bj->ij', self.cov_weights,
                                sigma_points - mean_hat, error_y)
        cov_y_y = jnp.einsum('b, bi, bj->ij', self.cov_weights, error_y, error_y) \
            + self.sigma_analysis
        
        temp = jnp.dot(cov_theta_y, jnp.linalg.inv(cov_y_y))
        mean_next = mean_hat + jnp.dot(obs - y_hat_mean, temp.T)
        cov_next = cov_hat - jnp.dot(temp, cov_theta_y.T)
        return mean_next.squeeze(), cov_next

    def sample(self, forward, init_mean, obs, N_iter,true_forward = None,
               update_freq = 0, init_cov = None,
               unscented_transform = 'modified-2n+1'):
        """
        forward: the forward operator
        init_mean: the inital mean vector for the parameters
        obs: the observation vector
        obs_cov: the noise matrix
        N_iter: the number of uki step
        update_freq: the frequency for updating the different matrix
        unscented_transform: the type of unscented transform, including modified-2n+1
        original-2n+1, modified-n+2, original-n+2
        model_error_mean: the approximate model error obtained by deeponet model 
        model_error_cov: the approximate cov obtained by deeponet model 
        """ 
        
        self.get_weights(unscented_transform)
        self.init_mean = init_mean
        self.mean = [init_mean]
        if init_cov is None:
            init_cov = self.init_cov
        self.cov = [init_cov]
        regulizer = (self.delta_t/(1 - self.delta_t) + 1 - self.alpha**2)
        self.sigma_analysis = (1/self.delta_t) * jnp.eye(self.obs_dim)*self.sigma**2
        self.sigma_predict = regulizer * self.init_cov
        pbar = trange(N_iter)
        for i in pbar:
            if update_freq > 0 and (i + 1) % update_freq == 0:
                self.sigma_predict = regulizer * init_cov
            init_mean, init_cov = self.analysis(forward, init_mean, init_cov, obs) 
            self.mean.append(init_mean)
            self.cov.append(init_cov)
                
        self.mean = jnp.vstack(self.mean)
        true_y = true_forward(self.mean)
        predict_y = forward(self.mean)
        self.error = jnp.linalg.norm((true_y - obs)/self.sigma_analysis[0,0], axis = 1)/2
        self.model_error = jnp.linalg.norm(true_y - predict_y, axis = 1)/jnp.linalg.norm(true_y, axis = 1)
        self.error = self.error.squeeze()
        self.model_error = self.model_error.squeeze()
        # index = jnp.nanargmin(self.error)
        # print(self.error)
        # print("small_error: {}, index: {}".format(self.error[index], index))
        # self.Mean = self.mean
        # self.Cov = self.cov
        # self.Error = self.error
        
        # self.error = self.error[:index+1]
        # self.mean = self.mean[:index+1]
        # self.cov = self.cov[:index + 1]
                    
        return init_mean, init_cov

    def get_weights(self, transform):
        """This generates the weights for the sigma points"""
        if transform in ['modified-2n+1', 'original-2n+1']:
            #ensemble size
            N_ens = self.para_dim * 2 + 1
            mean_weights = jnp.zeros(N_ens)
            cov_weights = jnp.zeros(N_ens)
            kappa, beta = 0.0, 2.0
            alpha = min(jnp.sqrt(4/(self.para_dim + kappa)), 1.0)
            lam = alpha**2*(self.para_dim + kappa) - self.para_dim
            c_weights = jnp.sqrt(self.para_dim + lam)*jnp.ones(self.para_dim)

            mean_weights = mean_weights.at[0].set(lam/(self.para_dim + lam))
            mean_weights = mean_weights.at[1:].set(1/(2*(self.para_dim + lam)))
            cov_weights = cov_weights.at[0].set(lam/(self.para_dim + lam) + 1 - alpha**2 + beta)
            cov_weights = cov_weights.at[1:].set(1/(2*(self.para_dim + lam)))
            
            if transform == 'modified-2n+1':
                mean_weights = mean_weights.at[0].set(1)
                mean_weights = mean_weights.at[1:].set(0)
                
        elif transform in ['original-n+2', 'modified-n+2']:
            N_ens = self.para_dim + 2
            mean_weights = jnp.zeros(N_ens)
            cov_weights = jnp.zeros(N_ens)
            c_weights = jnp.zeros((self.para_dim, N_ens))
            alpha = self.para_dim/(4*(self.para_dim + 1))
            IM = jnp.zeros((self.para_dim, self.para_dim + 1))
            IM = IM.at[0].set(jnp.array([-1,1])*jnp.sqrt(2*alpha))
            for i in range(1, self.para_dim):
                for j in range(i):
                    IM = IM.at[i,j].set(1/jnp.sqrt(alpha*i*(i+1)))
                IM = IM.at[i, i+1].set(-i/jnp.sqrt(alpha*i*(i+1)))
            c_weights = c_weights.at[:, 1:].set(IM)
            
            if transform == 'oringinal-n+2':
                mean_weights = 1/(self.para_dim + 1)
                mean_weights = mean_weights.at[0].set(0)
                cov_weights = alpha 
                cov_weights = cov_weights.at[0].set(0)
            else:
                mean_weights = 0
                mean_weights = mean_weights.at[0].set(1)
                cov_weights = alpha 
                cov_weights = cov_weights.at[0].set(0)
        self.c_weights = c_weights 
        self.mean_weights = mean_weights
        self.cov_weights = cov_weights
        self.num_sigma = N_ens

class UKI:
    """This class implements uki"""
    def __init__(self, para_dim, obs_dim, gamma, sigma, alpha, delta_t = 0.5) -> None:
        """
            Para:
                para_dim: the dimension of the parameter space
                obs_dim: the dimensionl of the observation spave
                gamma: the parameter for the initial covariance
                alpha: the regularization parameter
                
        """
        self.para_dim = para_dim
        self.obs_dim = obs_dim
        self.alpha = alpha 
        self.delta_t = delta_t
        self.sigma = sigma
        self.init_cov = jnp.eye(para_dim)*gamma**2
      
        
        
         

    def get_sigma_points(self, mean, cov):
        """This file constructs the sigma points"""
        R = sqrtm(cov)
        points = jnp.zeros((self.num_sigma, self.para_dim))
        if len(self.c_weights.shape) == 1:
            points = points.at[0].set(mean)
            temp = jnp.einsum('i, ij->ij', self.c_weights, R.T)
            points = points.at[1:self.para_dim + 1].set(mean + temp)
            points = points.at[self.para_dim + 1:].set(mean - temp)
        elif len(self.c_weights.shape) == 2:
            points = points.at[0].set(mean) 
            points = points.at[1:].set(mean + jnp.dot(R, self.c_weights))
        return points 

    def prediction(self, mean, cov):
        """This function constructs the prediction step"""
        mean_hat = self.alpha * mean + (1 - self.alpha) * self.init_mean
        cov_hat = self.alpha**2 * cov + self.sigma_predict
        return mean_hat, cov_hat 

    def analysis(self, forward, mean, cov, obs):
        ###prediction step
        mean_hat, cov_hat = self.prediction(mean, cov)
        ###construct ensembles
        sigma_points = self.get_sigma_points(mean_hat, cov_hat)
        y_hat = forward(sigma_points).reshape(self.num_sigma, -1)
        y_hat_mean = jnp.einsum('j, ji->i', self.mean_weights, y_hat)
        ###construct error
        error_y = y_hat - y_hat_mean
        cov_theta_y = jnp.einsum('b, bi, bj->ij', self.cov_weights,
                                sigma_points - mean_hat, error_y)
        cov_y_y = jnp.einsum('b, bi, bj->ij', self.cov_weights, error_y, error_y) \
            + self.sigma_analysis
        
        temp = jnp.dot(cov_theta_y, jnp.linalg.inv(cov_y_y))
        mean_next = mean_hat + jnp.dot(obs - y_hat_mean, temp.T)
        cov_next = cov_hat - jnp.dot(temp, cov_theta_y.T)
        return mean_next.squeeze(), cov_next

    def sample(self, forward, init_mean, obs, N_iter,true_forward = None,
               update_freq = 0, init_cov = None,
               unscented_transform = 'modified-2n+1'):
        """
        forward: the forward operator
        init_mean: the inital mean vector for the parameters
        obs: the observation vector
        obs_cov: the noise matrix
        N_iter: the number of uki step
        update_freq: the frequency for updating the different matrix
        unscented_transform: the type of unscented transform, including modified-2n+1
        original-2n+1, modified-n+2, original-n+2
        model_error_mean: the approximate model error obtained by deeponet model 
        model_error_cov: the approximate cov obtained by deeponet model 
        """ 
        
        self.get_weights(unscented_transform)
        self.init_mean = init_mean
        self.mean = [init_mean]
        if init_cov is None:
            init_cov = self.init_cov
        self.cov = [init_cov]
        regulizer = (self.delta_t/(1 - self.delta_t) + 1 - self.alpha**2)
        self.sigma_analysis = (1/self.delta_t) * jnp.eye(self.obs_dim)*self.sigma**2
        self.sigma_predict = regulizer * self.init_cov
        pbar = trange(N_iter)
        for i in pbar:
            if update_freq > 0 and (i + 1) % update_freq == 0:
                self.sigma_predict = regulizer * init_cov
            init_mean, init_cov = self.analysis(forward, init_mean, init_cov, obs) 
            self.mean.append(init_mean)
            self.cov.append(init_cov)
        
        start = time.time()     
        self.mean = jnp.vstack(self.mean)
        true_y = true_forward(self.mean)
        predict_y = forward(self.mean)
        self.predict_error = jnp.linalg.norm((predict_y - obs)/self.sigma_analysis[0,0], axis = 1)/2
        self.error = jnp.linalg.norm((true_y - obs)/self.sigma_analysis[0,0], axis = 1)/2
        self.error = self.error.squeeze()
        index = jnp.nanargmin(self.error)
        print(self.error)
        print("small_error: {}, index: {}".format(self.error[index], index))
        self.Mean = self.mean
        self.Cov = self.cov
        self.Error = self.error
        self.Predict_error = self.predict_error
        
        self.error = self.error[:index+1]
        self.predict_error = self.error[:index+1]
        self.mean = self.mean[:index+1]
        self.cov = self.cov[:index + 1]
        self.index = index 
        print(time.time() - start)
                    
        return self.mean[index], self.cov[index]

    def get_weights(self, transform):
        """This generates the weights for the sigma points"""
        if transform in ['modified-2n+1', 'original-2n+1']:
            #ensemble size
            N_ens = self.para_dim * 2 + 1
            mean_weights = jnp.zeros(N_ens)
            cov_weights = jnp.zeros(N_ens)
            kappa, beta = 0.0, 2.0
            alpha = min(jnp.sqrt(4/(self.para_dim + kappa)), 1.0)
            lam = alpha**2*(self.para_dim + kappa) - self.para_dim
            c_weights = jnp.sqrt(self.para_dim + lam)*jnp.ones(self.para_dim)

            mean_weights = mean_weights.at[0].set(lam/(self.para_dim + lam))
            mean_weights = mean_weights.at[1:].set(1/(2*(self.para_dim + lam)))
            cov_weights = cov_weights.at[0].set(lam/(self.para_dim + lam) + 1 - alpha**2 + beta)
            cov_weights = cov_weights.at[1:].set(1/(2*(self.para_dim + lam)))
            
            if transform == 'modified-2n+1':
                mean_weights = mean_weights.at[0].set(1)
                mean_weights = mean_weights.at[1:].set(0)
                
        elif transform in ['original-n+2', 'modified-n+2']:
            N_ens = self.para_dim + 2
            mean_weights = jnp.zeros(N_ens)
            cov_weights = jnp.zeros(N_ens)
            c_weights = jnp.zeros((self.para_dim, N_ens))
            alpha = self.para_dim/(4*(self.para_dim + 1))
            IM = jnp.zeros((self.para_dim, self.para_dim + 1))
            IM = IM.at[0].set(jnp.array([-1,1])*jnp.sqrt(2*alpha))
            for i in range(1, self.para_dim):
                for j in range(i):
                    IM = IM.at[i,j].set(1/jnp.sqrt(alpha*i*(i+1)))
                IM = IM.at[i, i+1].set(-i/jnp.sqrt(alpha*i*(i+1)))
            c_weights = c_weights.at[:, 1:].set(IM)
            
            if transform == 'oringinal-n+2':
                mean_weights = 1/(self.para_dim + 1)
                mean_weights = mean_weights.at[0].set(0)
                cov_weights = alpha 
                cov_weights = cov_weights.at[0].set(0)
            else:
                mean_weights = 0
                mean_weights = mean_weights.at[0].set(1)
                cov_weights = alpha 
                cov_weights = cov_weights.at[0].set(0)
        self.c_weights = c_weights 
        self.mean_weights = mean_weights
        self.cov_weights = cov_weights
        self.num_sigma = N_ens

class UKI_np:
    """This class implements uki"""
    def __init__(self, para_dim, obs_dim, gamma, sigma, alpha, delta_t = 0.5) -> None:
        """
            Para:
                para_dim: the dimension of the parameter space
                obs_dim: the dimensionl of the observation spave
                gamma: the parameter for the initial covariance
                alpha: the regularization parameter
                
        """
        self.para_dim = para_dim
        self.obs_dim = obs_dim
        self.alpha = alpha 
        self.delta_t = delta_t
        self.sigma = sigma
        self.init_cov = np.eye(para_dim)*gamma**2
      
        
        
         

    def get_sigma_points(self, mean, cov):
        """This file constructs the sigma points"""
        R = sqrtm(cov)
        points = np.zeros((self.num_sigma, self.para_dim))
        if len(self.c_weights.shape) == 1:
            points[0] = mean
            temp = np.einsum('i, ij->ij', self.c_weights, R.T)
            points[1:self.para_dim + 1] = mean + temp
            points[self.para_dim + 1:] = mean - temp
        elif len(self.c_weights.shape) == 2:
            points[0] = mean 
            points[1:] = mean + jnp.dot(R, self.c_weights)
        return points 

    def prediction(self, mean, cov):
        """This function constructs the prediction step"""
        mean_hat = self.alpha * mean + (1 - self.alpha) * self.init_mean
        cov_hat = self.alpha**2 * cov + self.sigma_predict
        return mean_hat, cov_hat 

    def analysis(self, forward, mean, cov, obs):
        ###prediction step
        mean_hat, cov_hat = self.prediction(mean, cov)
        ###construct ensembles
        sigma_points = self.get_sigma_points(mean_hat, cov_hat)
        y_hat = forward(sigma_points).reshape(self.num_sigma, -1)
        y_hat_mean = np.einsum('j, ji->i', self.mean_weights, y_hat)
        ###construct error
        error_y = y_hat - y_hat_mean
        cov_theta_y = np.einsum('b, bi, bj->ij', self.cov_weights,
                                sigma_points - mean_hat, error_y)
        cov_y_y = np.einsum('b, bi, bj->ij', self.cov_weights, error_y, error_y) \
            + self.sigma_analysis
        
        temp = np.dot(cov_theta_y, np.linalg.inv(cov_y_y))
        mean_next = mean_hat + np.dot(obs - y_hat_mean, temp.T)
        cov_next = cov_hat - np.dot(temp, cov_theta_y.T)
        return mean_next.squeeze(), cov_next

    def sample(self, forward, init_mean, obs, N_iter,true_forward = None,
               update_freq = 0, init_cov = None,
               unscented_transform = 'modified-2n+1'):
        """
        forward: the forward operator
        init_mean: the inital mean vector for the parameters
        obs: the observation vector
        obs_cov: the noise matrix
        N_iter: the number of uki step
        update_freq: the frequency for updating the different matrix
        unscented_transform: the type of unscented transform, including modified-2n+1
        original-2n+1, modified-n+2, original-n+2
        model_error_mean: the approximate model error obtained by deeponet model 
        model_error_cov: the approximate cov obtained by deeponet model 
        """ 
        
        self.get_weights(unscented_transform)
        self.init_mean = init_mean
        self.mean = [init_mean]
        if init_cov is None:
            init_cov = self.init_cov
        self.cov = [init_cov]
        regulizer = (self.delta_t/(1 - self.delta_t) + 1 - self.alpha**2)
        self.sigma_analysis = (1/self.delta_t) * np.eye(self.obs_dim)*self.sigma**2
        self.sigma_predict = regulizer * self.init_cov
        pbar = trange(N_iter)
        for i in pbar:
            if update_freq > 0 and (i + 1) % update_freq == 0:
                self.sigma_predict = regulizer * init_cov
            init_mean, init_cov = self.analysis(forward, init_mean, init_cov, obs) 
            self.mean.append(init_mean)
            self.cov.append(init_cov)
                
        self.mean = np.vstack(self.mean)
        true_y = true_forward(self.mean)
        predict_y = forward(self.mean)
        self.predict_error = np.linalg.norm((predict_y - obs)/self.sigma_analysis[0,0], axis = 1)/2
        self.error = np.linalg.norm((true_y - obs)/self.sigma_analysis[0,0], axis = 1)/2
        self.error = self.error.squeeze()
        index = np.nanargmin(self.error)
        print(self.error)
        print("small_error: {}, index: {}".format(self.error[index], index))
        self.Mean = self.mean
        self.Cov = self.cov
        self.Error = self.error
        self.Predict_error = self.predict_error
        
        self.error = self.error[:index+1]
        self.predict_error = self.error[:index+1]
        self.mean = self.mean[:index+1]
        self.cov = self.cov[:index + 1]
        self.index = index 
                    
        return self.mean[index], self.cov[index]

    def get_weights(self, transform):
        """This generates the weights for the sigma points"""
        if transform in ['modified-2n+1', 'original-2n+1']:
            #ensemble size
            N_ens = self.para_dim * 2 + 1
            mean_weights = np.zeros(N_ens)
            cov_weights = np.zeros(N_ens)
            kappa, beta = 0.0, 2.0
            alpha = min(jnp.sqrt(4/(self.para_dim + kappa)), 1.0)
            lam = alpha**2*(self.para_dim + kappa) - self.para_dim
            c_weights = jnp.sqrt(self.para_dim + lam)*jnp.ones(self.para_dim)

            mean_weights[0] = lam/(self.para_dim + lam)
            mean_weights[1:] = 1/(2*(self.para_dim + lam))
            cov_weights[0] = lam/(self.para_dim + lam) + 1 - alpha**2 + beta
            cov_weights[1:] = 1/(2*(self.para_dim + lam))
            
            if transform == 'modified-2n+1':
                mean_weights[0] = 1
                mean_weights[1:] = 0
                
        elif transform in ['original-n+2', 'modified-n+2']:
            N_ens = self.para_dim + 2
            mean_weights = np.zeros(N_ens)
            cov_weights = np.zeros(N_ens)
            c_weights = np.zeros((self.para_dim, N_ens))
            alpha = self.para_dim/(4*(self.para_dim + 1))
            IM = np.zeros((self.para_dim, self.para_dim + 1))
            IM[0] = np.array([-1,1])*np.sqrt(2*alpha)
            for i in range(1, self.para_dim):
                for j in range(i):
                    IM[i,j] = 1/jnp.sqrt(alpha*i*(i+1))
                IM[i, i+1] = -i/jnp.sqrt(alpha*i*(i+1))
            c_weights = c_weights.at[:, 1:].set(IM)
            
            if transform == 'oringinal-n+2':
                mean_weights = 1/(self.para_dim + 1)
                mean_weights[0] = 0
                cov_weights = alpha 
                cov_weights[0] = 0
            else:
                mean_weights = 0
                mean_weights[0] = 1
                cov_weights = alpha 
                cov_weights[0] = 0
        self.c_weights = c_weights 
        self.mean_weights = mean_weights
        self.cov_weights = cov_weights
        self.num_sigma = N_ens

class MCMC:
    """This class implements mcmc class sampling methods including pcn and mh"""
    def __init__(self, para_dim) -> None:
        """
           para:
               This is the parameter dimension
        """
        self.para_dim = para_dim 
    
    def MH(self, log_posterior, init_sample, num_samples, lr):
        current = init_sample
        accept = 0
        accept_samples = np.zeros((num_samples, self.para_dim))
        pbar = tqdm(range(num_samples))
        for i in pbar:
            proposed = np.random.randn(1, self.para_dim) * lr + current
            acc = np.min(log_posterior(proposed) - log_posterior(current), 0)
            pbar.set_postfix_str(f'Accept: {accept}/{num_samples}, Acc: %.3f'%(acc))
            if np.random.rand(1) < np.exp(acc):
                accept_samples[accept] = proposed.squeeze()
                current = proposed
                accept += 1
            else:
                current = current
        return accept_samples[:accept]
    
    def pcn(self, log_likelihood, init_sample, num_samples, beta):
        current = init_sample
        accept = 0
        accept_samples = np.zeros((num_samples, self.para_dim))
        pbar = tqdm(range(num_samples))
        for i in pbar:
            proposed = beta * np.random.randn(1, self.para_dim) + np.sqrt(1 - beta**2) * current
            acc = np.min(log_likelihood(proposed) - log_likelihood(current), 0)
            if np.random.rand(1) < np.exp(acc):
                accept_samples[accept] = proposed.squeeze()
                current = proposed
                accept += 1
                pbar.set_postfix_str(f'Accept: {accept}/{num_samples}')
            else:
                current = current

        return accept_samples[:accept] 

class EKI:
    """This class implements ensemble kalman sampling"""
    def __init__(self, para_dim, obs_dim, sigma_noise, gamma, alpha) -> None:
        """
            Para:
                para_dim: the dimension of the parameter space
                obs_dim: the dimensionl of the observation spave
                sigma_noise: the noise sigma matrix
                gamma: the parameter
        """
        self.para_dim = para_dim
        self.alpha = alpha
        self.sigma_analysis = sigma_noise ** 2 * np.eye(obs_dim)
        self.sigma_predict = np.eye(para_dim) * (2 - self.alpha**2) * gamma 
        self.sigma_analysis_R = np.linalg.cholesky(self.sigma_analysis)
        self.sigma_predict_R = np.linalg.cholesky(self.sigma_predict)
        self.gamma = gamma
        self.obs_dim = obs_dim
    
    def predict(self, sigma_points):
        w_next = np.random.randn(self.num_samples, self.para_dim)@self.sigma_predict_R.T 
        theta_next = self.alpha * sigma_points + (1 - self.alpha)*self.init_mean + w_next
        mean_next = np.mean(theta_next, axis = 0)
        return mean_next
    
    def analysis(self, forward_operator, theta, y_obs):
        y_hat = forward_operator(theta)
        y_mean = np.mean(y_hat, axis = 0)
        mean_hat = self.predict(theta)
        cov_theta_y = np.einsum('bi, bj->ij', theta - mean_hat, y_hat - y_mean)
        cov_y_y = np.einsum('bi, bj->ij', y_hat - y_mean, y_hat - y_mean)
        cov_theta_y = cov_theta_y/(self.num_samples - 1)
        cov_y_y = cov_y_y/(self.num_samples - 1) + self.sigma_analysis
        v_hat = np.random.randn(self.num_samples, self.obs_dim)@self.sigma_analysis_R.T
        temp = np.dot(cov_theta_y, np.linalg.inv(cov_y_y))
        theta_next = theta + np.dot(temp, y_obs - y_hat - v_hat)
        mean_next = np.mean(theta_next, axis = 0)
        return theta_next, mean_next
        

    def sample(self, forward, num_step, num_samples, init_mean, y_obs):
        self.num_samples = num_samples
        self.mean = np.zeros((num_step, self.para_dim))
        self.init_mean = init_mean
        init_cov = self.gamma**2*np.eye(self.para_dim)
        init_cov_R = np.linalg.cholesky(init_cov)
        theta = np.random.randn(self.num_samples, self.para_dim)@init_cov_R.T
        for i in tqdm(range(num_step)):
            theta, init_mean = self.analysis(forward, theta, y_obs)
            self.mean[i] = init_mean
        return init_mean

        
            












