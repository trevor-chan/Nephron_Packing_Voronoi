import sys
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy
import scipy.special
import cv2

import torch

import networkx as nx

import pickle
from PIL import Image, ImageDraw
import matplotlib.lines as lines
from tqdm import tqdm
import glob

import statistics 

#For network adjacency checks
from skimage.transform import rotate
from sklearn.neighbors import KDTree
from scipy.spatial import distance

#For PCA

import warnings

from sys import argv

np.seterr(all='raise')
scipy.special.seterr(all='raise')

from lineage_timeless import lineage_timeless
from network_object import network_object
from cell_object import cell_object


'''
//  MAIN  //
'''
        
        
def main(): 
    assert len(argv) > 1, "(lineage directory) (rerun = False) (output location and filename - '.lineage')"

    lineage_path = argv[1]
    rerun = False
    if len(argv) > 2:
        rerun = True
    
    #if both output files exist, exit
    if os.path.isfile(lineage_path+".lineage") and rerun == False:
        sys.exit(lineage_path+".lineage  already exists")
    
    out_lineage = lineage_timeless(lineage_path, lineage_path, calc_fdim = True)
    
    if len(argv) >3:
        out_lineage.save_object(fout = argv[3] + ".lineage")
    else:
        out_lineage.save_object(fout = lineage_path +".lineage")
    

if __name__=="__main__": 
    main() 

