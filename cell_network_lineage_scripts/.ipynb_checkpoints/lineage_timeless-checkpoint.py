import sys
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy
import scipy.special
#import cv2

import torch

#import networkx as nx

import pickle
#from PIL import Image, ImageDraw
#import matplotlib.lines as lines
from tqdm import tqdm
import glob

import statistics 

#For network adjacency checks
#from skimage.transform import rotate
#from sklearn.neighbors import KDTree
#from scipy.spatial import distance

#For PCA
#import pandas as pd
#from sklearn.decomposition import PCA
#from sklearn.preprocessing import StandardScaler

#import seaborn as sns

import warnings

np.seterr(all='raise')
scipy.special.seterr(all='raise')

from network_object import network_object

class lineage_timeless:
    def __init__(self, filepath, tag, read = True, calc_fdim = False):
        if read == True:
            self.items = self.read_in(filepath, calc_fdim)
        self.tag = tag
        
    def read_in(self, filepath, calc_fdim):
        result = []
        for instance_path in tqdm(sorted(glob.glob(filepath+'/**/*.data',recursive = True)), desc = filepath, position=0, leave=True):
            instances = {}
            with open(instance_path, 'rb') as filehandle:
                try:
                    instances = pickle.load(filehandle)
                except RuntimeError:
                    #instances = torch.load(filehandle, pickle_module=pickle, map_location=torch.device('cpu'))
                    print("load failed for "+instance_path)
                    continue
            temp_network = network_object(instances, instance_path)
            if calc_fdim == True:
                temp_network.fractal_dim = temp_network.calc_fractal_dim()[0]
            result.append(temp_network)
        return result
    
    ## to be called from lineage: plots instance power exp to density
    def plot_fit_params(self, ax1in = None, ax2in = None):

        density = []
        popt = []
        Rsquared = []

        for network in self.items:
            try:
                popt_temp, Rsquared_temp, RMSE, pcov, perr, xdata, ydata = network.fit_power()
            except RuntimeError:
                popt_temp = (float('nan'),float('nan'),float('nan'))
                Rsquared_temp = float('nan')
            
            if Rsquared_temp==1:
                continue
            popt.append(popt_temp)
            Rsquared.append(Rsquared_temp)
            density.append(network.density)

        if ax1in is not None and ax2in is not None:
            ax1=ax1in
            ax2=ax2in
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        ax1.plot(density,Rsquared,'.',label = self.tag)
        ax1.set_title("Rsquared to density")
        ax2.plot(density,[param[1] for param in popt],'.',label = self.tag)
        ax2.set_title("critical exponent to density")
        
        #plt.suptitle(lineage.tag + '_' + mindist)
        #plt.savefig(output+'_'+mindist+'.png')
        #plt.show()
    
    def load_object(self, fin = "lineage.lineage"):
        f = open(fin, 'rb')
        tmp_dict = pickle.load(f)
        f.close()
        self.__dict__.update(tmp_dict) 

    def save_object(self, fout = "lineage.lineage"):
        f = open(fout, 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()
        
        
            