import os
import gif
import numpy as np
import seaborn as sns 
from typing import List
import matplotlib.pyplot as plt 
from scipy.interpolate import interpn
from mpl_toolkits.axes_grid1 import make_axes_locatable



__all__ = ['Plotters']

class Plotters:
    """This class implements plotters functions"""
    def __init__(self, img_save_path) -> None:
        self.img_save_path = img_save_path

        if not os.path.exists(img_save_path):
            os.makedirs(img_save_path)
        
    def plot_comparison_2d(self, true_y, predict_y, titles, prefix, title = None, mesh = None, save = False):
        true_y = true_y.flatten()
        dim = np.sqrt(max(true_y.shape)).astype(np.int32)
        plt.style.use('default')
        fig, ax = plt.subplots(1, 3)
        fig.set_figwidth(8)
        fig.set_figheight(2)
        true_y = true_y.reshape(dim, dim)
        predict_y = predict_y.reshape(dim, dim)
        err = abs(true_y - predict_y)
        if mesh == None:
            im1 = ax[0].imshow(true_y, cmap = 'jet', interpolation='bilinear')
            im2 = ax[1].imshow(predict_y, cmap = 'jet', interpolation='bilinear')
            im3 = ax[2].imshow(err, cmap = 'jet', interpolation='bilinear')
        else:
            true_y = true_y.reshape(mesh[0].shape)
            predict_y = predict_y.reshape(mesh[0].shape)
            err = abs(true_y - predict_y)
            im1 = ax[0].pcolor(mesh[0], mesh[1], true_y, cmap = 'jet')
            im2 = ax[1].pcolor(mesh[0], mesh[1], predict_y, cmap = 'jet')
            im3 = ax[2].pcolor(mesh[0], mesh[1], err, cmap = 'jet')
        ax[0].set_title(titles[0])
        ax[1].set_title(titles[1])
        ax[2].set_title(titles[2])
        im = [im1, im2, im3]
        for i in range(len(im)):
            divider1 = make_axes_locatable(ax[i])
            cax1 = divider1.append_axes('right', size = '2%', pad = 0.08)
            fig.colorbar(im[i], cax=cax1, orientation='vertical')
        plt.subplots_adjust(wspace=0.6)
        fig.set_dpi(150)
        plt.tight_layout()
        plt.title(title)
        if save:
            plt.savefig(os.path.join(self.img_save_path, prefix))
    

    def plot_comparison_1d(self, true_y, predict_y, title, prefix):
        plt.style.use('default')
        fig, ax = plt.subplots(1,3)
        fig.set_figwidth(8)
        fig.set_figheight(2)
        ax[0].plot(true_y.flatten())
        ax[1].plot(predict_y.flatten())
        ax[2].plot(true_y.flatten() - predict_y.flatten())
        ax[0].set_title(title[0])
        ax[1].set_title(title[1])
        ax[2].set_title(title[2])
        fig.set_dpi(150)
        plt.tight_layout()
        plt.savefig(os.path.join(self.img_save_path, prefix))

    
    def plot_comparison_gif(self, mean, true_mean, prefix, titles = None):
        if titles is None:
            titles = ['True', 'Predict', 'Absolute_error']
        gif.options.matplotlib['dpi'] = 200

        @gif.frame
        def plot(i):
            self.plot_comparison_2d(true_mean, mean[i], titles, prefix = str(i) + 'th_mean', title = str(i))
        
        frames = []
        for i in range(mean.shape[0]):
            frame = plot(i)
            frames.append(frame)
        
        gif.save(frames, os.path.join(self.img_save_path, prefix), duration = 600)
    
    def plot_train_data(self, res_points, initial_points, boundary_points, prefix):
        plt.style.use('ggplot')
        dim = res_points.shape[1]
        fig = plt.figure()
        if dim == 3:
            ax = fig.add_subplot(projection = '3d')
        else:
            ax = fig.add_subplot()
        ax.scatter(*res_points.T, label = "Residual points")
        ax.scatter(*boundary_points.T, label = 'Boundary points')
        ax.scatter(*initial_points.T, label = 'Initial points')
        ax.legend()
        # plt.show()
        plt.savefig(os.path.join(self.img_save_path, prefix))
    
    def plot_pair_distribution(self, predict, true, prefix = None):
        dimension = predict.shape[1]
        plt.style.use('default')
        fig, ax = plt.subplots(dimension, dimension)
        fig.set_dpi(100)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        
        legends = []
        for i in range(dimension):
            for j in range(dimension):
                if i < j:
                    ax[i, j].axis('off')
                elif i > j:
                    line1 = sns.kdeplot(x = predict[:, i], y = predict[:, j], ax = ax[i, j], color = 'red')
                    line2 = sns.kdeplot(x = true[:, i], y = predict[:, j], ax = ax[i, j], color = 'blue')
                else:
                    line1 = sns.kdeplot(x = predict[:, j], ax = ax[i, j], color = 'red')
                    line2 = sns.kdeplot(x = true[:, i], ax = ax[i, j], color = 'blue')
                    ax[i, j].set_ylabel('')
                ax[i, j].spines['right'].set_visible(False)
                ax[i, j].spines['top'].set_visible(False)
                ax[i, 0].set_ylabel('$\kappa_{}$'.format(i+1))
                ax[-1, j].set_xlabel('$\kappa_{}$'.format(j+1))
                legends.append(line1)
                legends.append(line2)
        fig.legend(legends, labels = ['Predict', 'True'], bbox_to_anchor=(0.8, 0.8), loc='center')
        plt.tight_layout()   
        plt.savefig(os.path.join(self.img_save_path, prefix))
    
    def plot_obs(self, true_y, obs, mesh, obs_mesh, title, prefix):
        plt.figure()
        if not isinstance(mesh, List):
            true_y = true_y.squeeze()
            obs = obs.squeeze()
            plt.plot(mesh, true_y, label = 'True_y')
            plt.scatter(obs_mesh, obs, label = 'Observations')
            plt.title(title)
            plt.legend()
        else:
            plt.pcolor(mesh[0], mesh[0], true_y, cmap = 'jet')
            plt.scatter(obs_mesh[0], obs_mesh[1], label = 'Observations')
        plt.tight_layout()
        plt.savefig(os.path.join(self.img_save_path, prefix))
            
        
            
        
            



