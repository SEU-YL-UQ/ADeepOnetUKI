# Adaptive Operator Learning for Infinite-Dimensional Bayesian Inverse Problems

This is the code for the paper "Adaptive Operator Learning for Infinite-Dimensional Bayesian Inverse Problems".

## Abstract
The fundamental computational issues in Bayesian inverse problems (BIPs) governed by partial differential equations (PDEs) stem from the requirement of repeated forward model evaluations. A popular strategy to reduce such costs is to replace expensive model simulations with computationally efficient approximations using operator learning, motivated by recent progress in deep learning. However, using the approximated model directly may introduce a modeling error, exacerbating the already ill-posedness of inverse problems. Thus, balancing between accuracy and efficiency is essential for the effective implementation of such approaches. To this end, we develop an adaptive operator learning framework that can reduce modeling error gradually by forcing the surrogate to be accurate in local areas. This is accomplished by adaptively fine-tuning the pretrained approximate model with training points chosen by a greedy algorithm during the posterior evaluation process. To validate our approach, we use DeepOnet to construct the surrogate and unscented Kalman inversion (UKI) to approximate the BIP solution, respectively. Furthermore, we present a rigorous convergence guarantee in the linear case using the UKI framework. The approach is tested on a number of benchmarks, including the Darcy flow, the heat source inversion problem, and the reaction-diffusion problem. The numerical results show that our method can significantly reduce computational costs while maintaining inversion accuracy.

## Citation
```
 @article{doi:10.1137/24M1643815,
author = {Gao, Zhiwei and Yan, Liang and Zhou, Tao},
title = {Adaptive Operator Learning for Infinite-Dimensional Bayesian Inverse Problems},
journal = {SIAM/ASA Journal on Uncertainty Quantification},
volume = {12},
number = {4},
pages = {1389-1423},
year = {2024},
doi = {10.1137/24M1643815}
}
```
