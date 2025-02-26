from jax import random
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt 
import os 
os.environ['CUDA_VISIBLE_DEVICES']='0'

from fenics import *
from functools import partial 
import warnings
warnings.filterwarnings("ignore")
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = 'false'
from models.train import DeepONet
from data_gen.GRF import GaussianRF_KL
from data_gen.data_generate import Data_generate
from data_gen.forward import Solver
from data_gen.dataloader import DataGenerator
from sampler.sampler import UKI
from utils.plotters import Plotters
from utils.choose import choose_high


class save_path:
    img_save_path = '/users/gzhiwei/scratch/python/test_case/experiments/flow2d/img'
    model_save_path = '/users/gzhiwei/scratch/python/test_case/experiments/flow2d/models'
    data_save_path = '/users/gzhiwei/scratch/python/test_case/experiments/flow2d/data'

if not os.path.exists(save_path.img_save_path):
    os.makedirs(save_path.img_save_path)
if not os.path.exists(save_path.data_save_path):
    os.makedirs(save_path.data_save_path)
if not os.path.exists(save_path.model_save_path):
    os.makedirs(save_path.model_save_path)

prior = GaussianRF_KL(3,2,1)
data = Data_generate()
num_kl = 128
plot = Plotters(save_path.img_save_path)
step = 10
discrete = 70
obs_dim = (discrete//step - 1)**2
shape = (discrete+1, discrete+1)
noise_level = 0.01

prior_mesh = UnitSquareMesh(discrete, discrete)
prior_V = FunctionSpace(prior_mesh, 'P', 1)
sample = partial(prior.sample, prior_V)
solver = lambda kappa: Solver.forward_flow(np.exp(kappa), prior_V)
obs_x = jnp.linspace(0, 1, discrete+1)[::step][1:-1]
obs_y = jnp.linspace(0, 1, discrete+1)[::step][1:-1]
obs_mesh = jnp.meshgrid(obs_x, obs_y)
obs_points = jnp.array([obs_mesh[0].flatten(), obs_mesh[1].flatten()]).T 


branch_layers = [num_kl] + 5*[100] + [64]
trunk_layers = [2] + 5*[100] + [64]
model = DeepONet(branch_layers, trunk_layers, save_path)

# base_key = random.PRNGKey(1234)
# randn_samples = random.normal(base_key, (1000, num_kl))
# if os.path.exists(os.path.join(save_path.data_save_path, 'training_data.npy')):
#     u_train, y_train, s_train = model.load_data(save_path.data_save_path, 'training_data')
# else:
#     u_train, y_train, s_train = data.generate_deeponet(
#         prior = prior, 
#         randn_samples = randn_samples,
#         forward_sover = solver,
#         prior_V = prior_V,
#         num_sensors = 2000
#     )
#     model.save_data([u_train, y_train, s_train], save_path.data_save_path, 'training_data')

# dataset = DataGenerator(u_train, y_train, s_train*100, batch_size=10000)




def true_forward(theta):
    theta = theta.reshape(-1, num_kl)
    samples = prior.sample(prior_V, theta).reshape(theta.shape[0], -1)
    solution = np.apply_along_axis(solver, arr = samples, axis = 1)
    solution = solution.reshape(-1, *shape)
    return jnp.asarray(solution[:, ::step, ::step][:, 1:-1, 1:-1].reshape(theta.shape[0], -1))

def predict_forward(theta):
    theta = theta.reshape(-1, num_kl)
    points = jnp.tile(obs_points, (theta.shape[0], 1))
    samples = jnp.repeat(theta, repeats = obs_points.shape[0], axis = 0)
    solution = model.predict_s(model.get_params(model.opt_state), samples, points)
    return solution.reshape(theta.shape[0], -1)

Tol = {'0.05':0.01, '0.1':0.01, '0.01':0.01}
num_adaptive = {'0.01':50, "0.05": 20, "0.1":20}
num_train = {"0.01":15000, "0.05":15000, "0.1":15000}

forward_fem = []
forward_dnn = []
model = DeepONet(branch_layers, trunk_layers, save_path)
if os.path.exists(os.path.join(save_path.model_save_path, 'checkpoint.npy')):
    model.load_model('checkpoint')
else:
    model.train(dataset, nIter = 100000)  

#the ground truth
ground_truth = random.uniform(random.PRNGKey(1234), (1,256), minval=-20, maxval=20)
true_kappa = prior.sample(prior_V, ground_truth)
true_y = solver(true_kappa).reshape(*shape)[::step, ::step][1:-1, 1:-1].flatten()
sigma = noise_level*true_y
np.random.seed(1)
obs = true_y + np.random.randn(1, obs_dim)*sigma

###the initial mean
init_mean = random.normal(random.PRNGKey(1234), (num_kl, ))
uki = UKI(num_kl, obs_dim, 1, 1, 1)
mean, cov = uki.sample(predict_forward, init_mean, obs, 20, true_forward)
direct_mean = uki.Mean

label = str(noise_level*100) +'.png'
model.save_data(direct_mean,save_path.data_save_path, 'kappa_direct' + label)
        
Mean = [uki.Mean]
Err = [uki.error]
Fitting_Err = [uki.Error]
count = [uki.index]

for i in range(10):
    sample_cov = jnp.diag(jnp.diag(cov))/jnp.max(cov)
    key = random.PRNGKey(i)
    total_samples = random.multivariate_normal(key, mean.reshape(1,-1), sample_cov, (2000, ))
    choose_samples = choose_high(mean, total_samples, predict_forward, num_adaptive[str(noise_level)])
    u_train, y_train, s_train = data.generate_deeponet(
        prior = prior,
        randn_samples = choose_samples,
        forward_sover = solver,
        prior_V = prior_V,
        num_sensors = 2000
    )
    dataset = DataGenerator(u_train, y_train, s_train, batch_size=10000)
    model.train(dataset, nIter = num_train[str(noise_level)], update = True)
    
    
    mean, cov = uki.sample(predict_forward, mean, obs, 10, true_forward, init_cov=cov)
                        
    tol = (Err[-1][-1] - uki.error[-1])/Err[-1][-1]
    

    if  0 < tol < Tol[str(noise_level)]:
        Mean.append(uki.Mean)
        Err.append(uki.error[1:])
        Fitting_Err.append(uki.Error)
        count.append(uki.index)
        break 
    elif tol >= Tol[str(noise_level)]:
        Mean.append(uki.Mean)
        Err.append(uki.error[1:])
        Fitting_Err.append(uki.Error)
        count.append(uki.index)
    elif tol <= 0:
        break

fem_mean, fem_cov = uki.sample(true_forward, init_mean, obs, 20, true_forward)
true_kappa = prior.sample(prior_V, ground_truth).reshape(1, -1)
fem_kappa = prior.sample(prior_V, uki.Mean).reshape(len(uki.Mean), -1)
dnn_kappa = prior.sample(prior_V, jnp.vstack(Mean)).reshape(len(jnp.vstack(Mean)), -1)
direct_kappa = prior.sample(prior_V, direct_mean).reshape(len(direct_mean), -1)
error_direct_kappa = jnp.linalg.norm(direct_kappa - true_kappa, axis = 1)/jnp.linalg.norm(true_kappa)
error_dnn_kappa = jnp.linalg.norm(dnn_kappa - true_kappa, axis = 1)/jnp.linalg.norm(true_kappa)
error_fem_kappa = jnp.linalg.norm(fem_kappa - true_kappa, axis = 1)/jnp.linalg.norm(true_kappa)

model.save_data(uki.Error, save_path.data_save_path, 'FEM_fitting_error' + label)
model.save_data(model.loss_log, save_path.data_save_path, 'loss_' + label) 
model.save_data(Mean, save_path.data_save_path, 'kappa_dnn' + label)
model.save_data(uki.mean,save_path.data_save_path, 'kappa_fem' + label)
model.save_data(error_dnn_kappa, save_path.data_save_path, 'error_kappa_dnn' + label)
model.save_data(error_fem_kappa, save_path.data_save_path, 'error_kappa_fem' + label)
model.save_data(error_direct_kappa, save_path.data_save_path, 'error_kappa_direct' + label)
model.save_data(true_kappa, save_path.data_save_path, 'true_kappa' + str(num))
model.save_data(Fitting_Err, save_path.data_save_path, 'DNN_fitting_error' + label)
model.save_data(count, save_path.data_save_path, 'count' + label)

forward_dnn.append((num_adaptive[str(noise_level)] + 20)*(i+1)) 
forward_fem.append(uki.num_sigma * (len(uki.mean) - 1) + len(uki.mean))      

        
        
        

