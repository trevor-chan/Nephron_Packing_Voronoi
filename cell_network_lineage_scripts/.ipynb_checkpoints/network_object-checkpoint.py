#Class object for network:
'''contains: the base image
the adjacency list
the graph
notable properties calculated on initialize
clustering (to be fixed) on call
visualization functions for graph on image and graph alone'''



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
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import warnings
from cell_object import cell_object

np.seterr(all='raise')
scipy.special.seterr(all='raise')




class network_object:
    #network object - takes in a (full) segmentation output and initializes a network-object with notable properties calculated
    
    def __init__(self, instances, tag, img = None):
        
        self.image = img
        self.instances = instances
        self.tag = tag
        
        #boxes = np.array(instances['pred_boxes'])
        #polygons = np.array(instances['pred_masks'])
        #scores = np.array(instances['scores'])
        
        self.number = len(self.instances['pred_boxes'])
        self.density = self.number/(instances['image_size'][0]*instances['image_size'][1])
        self.avgscore = sum(self.instances['scores'])/self.number
        
        self.densityHistogram = 0
        self.centroid_list = self.construct_centroid_list(self.instances)
        #self.polygon_list = self.construct_polygon_list(self.instances)
        
        self.adjacency_list = self.construct_adjacencies(self.instances)
        
        self.graph = self.construct_graph(self.adjacency_list)
        
        self.num_components = len(self.get_component_masses())
        
        self.fractal_dim = None
        
        cell_list = []
        for i in range(len(instances['pred_boxes'])):
            cell_list.append(cell_object(instances['pred_boxes'][i],instances['pred_masks'][i],instances['scores'][i]))
        
        def avg(x):
            return sum(x)/len(x)
        careas = [cell.area for cell in cell_list]
        self.cell_areas = avg(careas)
        self.cell_areas_var = np.var(careas)
        cperim = [cell.perimeter for cell in cell_list]
        self.cell_perimeters = avg(cperim)
        self.cell_perimeters_var = np.var(cperim)
        ccircul = [cell.circularity for cell in cell_list]
        self.cell_circularities = avg(ccircul)
        self.cell_circularities_var = np.var(ccircul)
        cmajax = [cell.majaxis for cell in cell_list]
        self.cell_majaxes = avg(cmajax)
        self.cell_majaxes_var = np.var(cmajax)
        cminax = [cell.minaxis for cell in cell_list]
        self.cell_minaxes = avg(cminax)
        self.cell_minaxes_var = np.var(cminax)
        casra = np.divide(cmajax,cminax)
        self.asra = avg(casra)
        self.asra_var = np.var(casra)
        
        self.cell_scores = avg([cell.score for cell in cell_list])
        
    def construct_centroid_list(self, instances):
        return np.array([ (((box[0]+box[2])/2).item() , ((box[1]+box[3])/2).item()) for box in instances["pred_boxes"] ])
    
    def construct_polygon_list(self, instances):
        return [ np.reshape(mask[0], (int(len(mask[0])/2) , 2)) for mask in instances["pred_masks"] ]
        
    #Definition of mindist parameter -------------------------------------------------------------------------------------
    def construct_adjacencies(self, instances, mindist = 10):
        #takes a list of n cell_poly objects
        #takes a minimum distance defined as adjacent
        #returns an adjacency list of length n

        adjacency_list = []
        wide_list = []

        centroid_list = [ (((box[0]+box[2])/2).item() , ((box[1]+box[3])/2).item()) for box in instances["pred_boxes"] ]
        centroid_list = np.array(centroid_list)
        polygon_list = [ np.reshape(mask[0], (int(len(mask[0])/2) , 2)) for mask in instances["pred_masks"] ]

        tree = []
        tree = KDTree(centroid_list, leaf_size=10)

        wide_list = tree.query_radius(centroid_list, mindist * 5)

        for i,small_list in enumerate(wide_list):
            adjacency_list.append (list())
            for index in small_list:
                # if we've already explored this pair then skip (as cdist _should_ be symmetrical)
                if index <= i:
                    continue
                d = distance.cdist(polygon_list[i],polygon_list[index],'euclidean')
                if min(d.flatten()) < mindist:
                    adjacency_list[i].append (index)

        return adjacency_list
        
    def construct_graph(self, adjacency_list):
        g = nx.Graph()
        for i in range(len(adjacency_list)):
            g.add_node(i)

        for a in range(len(adjacency_list)):
            for b in adjacency_list[a]:
                if g.has_edge(a,b):
                    continue
                g.add_edge(a,b)
        return g
    
    def average_degree(self):
        return sum([len(sublist) for sublist in self.adjacency_list])/self.number
                                                                         
    def degree_variance(self):
        numadj = [len(sublist) for sublist in self.adjacency_list]
        return statistics.variance(numadj)
        
    def plot_degree(self):
        num_adjacencies = [len(sublist) for sublist in self.adjacency_list]
        fig = plt.figure()
        plt.hist(num_adjacencies, max(num_adjacencies))
        fig.suptitle('Number of adjacent cells')
        plt.xlabel('# neighbors')
        plt.ylabel('count')
    
    def calc_fractal_dim(self, threshold = 0.25, point = False, image = True):
        from PIL import ImageDraw
        import matplotlib.patches as patches

        def fractal_dimension(Z, threshold=0.9):
            def boxcount(Z, k):
                S = np.add.reduceat(
                    np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                                       np.arange(0, Z.shape[1], k), axis=1)
                #return len(np.where((S > 0) & (S < k*k))[0])#--------------------(if S< k*k) will exclude filled portions,
                                                                                #not necessary for this application
                return len(np.where(S > 0)[0])
            Z = (Z < threshold)
            p = min(Z.shape)
            n = 2**np.floor(np.log(p)/np.log(2))
            n = int(np.log(n)/np.log(2))
            sizes = 2**np.arange(n, 1, -1)
            counts = []
            for size in sizes:
                counts.append(boxcount(Z, size))
            coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)

            return -coeffs[0]

        def binarize_img(instances):
            #calculate all masks
            bounding_boxes = np.array(instances['pred_boxes'])
            polygons = np.array(instances['pred_masks'])

            result = np.zeros((instances['image_size'][1],instances['image_size'][0]), dtype=int)

            for i in range(len(bounding_boxes)):
                polygon = np.copy(polygons[i])

                img = Image.new('1', instances['image_size'], 0)
                img1 = ImageDraw.Draw(img)
                img1.polygon(polygon[0].tolist(), fill = 1, outline = 1)
                result = np.add(result,img)
            return result

        def construct_centroid_list(instances):
            return np.array([ (((box[0]+box[2])/2).item() , ((box[1]+box[3])/2).item()) for box in instances["pred_boxes"] ])

        def binarize_centroids(instances):
            #calculate all masks
            bounding_boxes = np.array(instances['pred_boxes'])
            centroids = construct_centroid_list(instances)

            result = np.zeros((instances['image_size'][1],instances['image_size'][0]), dtype=int)


            for centroid in centroids:
                result[int(centroid[0]),int(centroid[1])] = 1
            return result


        inst = self.instances
        
        if point:
            ZZ = binarize_centroids(inst)
        if image:
            Z = binarize_img(inst)

        '''
        sizes = 128, 64, 32, 16  #, 8
        xmin, xmax = 0, Z.shape[1]
        ymin, ymax = 0, Z.shape[0]
        fig = plt.figure(figsize=(15, 10))

        for i, size in enumerate(sizes):
            ax = plt.subplot(1, len(sizes), i+1, frameon=False)
            ax.imshow(1-Z, plt.cm.gray, interpolation="bicubic", vmin=0, vmax=1,
                      extent=[xmin, xmax, ymin, ymax], origin="upper")
            ax.set_xticks([])
            ax.set_yticks([])
            for y in range(Z.shape[0]//size+1):
                for x in range(Z.shape[1]//size+1):
                    s = (Z[y*size:(y+1)*size, x*size:(x+1)*size] > 0.25).sum()
                    if s > 0 and s < size*size:
                        rect = patches.Rectangle(
                            (x*size, Z.shape[0]-1-(y+1)*size),
                            width=size, height=size,
                            linewidth=.5, edgecolor='.25',
                            facecolor='.75', alpha=.4)
                        ax.add_patch(rect)

        plt.tight_layout()
        plt.savefig("fractal-dimension.png", dpi = 150)
        plt.show()
        '''

        #returns tuple as (fractal dim binarized img, fractal dim binarized centroids)
        if point and image:
            return fractal_dimension(Z, threshold = threshold), fractal_dimension(ZZ, threshold = threshold)
        elif point:
            return 0, fractal_dimension(ZZ, threshold = threshold)
        else:
            return fractal_dimension(Z, threshold = threshold), 0
    
        
    def calc_chromatic(self):
        return max(nx.coloring.greedy_color(self.graph, strategy=nx.coloring.strategy_largest_first).values())
    
    def calc_avg_chromatic(self):
        temp = nx.coloring.greedy_color(self.graph, strategy=nx.coloring.strategy_largest_first).values()
        return sum(temp)/len(temp)
    
    def get_component_masses(self):
        components_g = nx.connected_components(self.graph)
        gc = []
        for component in components_g:
            gc.append(len(component))
        return sorted(gc)
    
    def get_max_comp_mass(self):
        return self.get_component_masses()[-1]
    
    def get_largest_comp_ratio(self):
        a = self.get_component_masses()
        return a[-1]/a[-2]
    
    
    def assortativity(self):
        try:
            return max(0,nx.degree_pearson_correlation_coefficient(self.graph))
        except:
            return float("NaN")
    
    
#Entropy Calculations
    
    def shannon_entropy(self, labels):
        """ Computes entropy of label distribution. this code from stack overflow user Jarad, 
        https://stackoverflow.com/questions/15450192/fastest-way-to-compute-entropy-in-python"""
        
        n_labels = len(labels)
        if n_labels <= 1:
            return 0
        value,counts = np.unique(labels, return_counts=True)
        probs = counts / n_labels
        n_classes = np.count_nonzero(probs)
        if n_classes <= 1:
            return 0
        ent = 0.
        # Compute entropy
        #base = 2 #Entropy calculation default to bits
        for i in probs:
            ent -= i * np.log2(i)
        return ent
    
    def degree_entropy(self):
        return self.shannon_entropy([len(sublist) for sublist in self.adjacency_list])
    
    def compmass_entropy(self):
        return self.shannon_entropy(self.get_component_masses())
    
    def colorability_entropy(self):
        return self.shannon_entropy(list(nx.coloring.greedy_color(self.graph, strategy=nx.coloring.strategy_largest_first).values()))
    
    
    
#Vizualizations
    
        
    def visualize_adjacencies(self, img = None, savefig = 0):
        fig, ax = plt.subplots()
        if self.image is None and img is None:
            return 0
        elif self.image is None:
            ax.imshow(img, cmap=plt.cm.gray)
        else:
            ax.imshow(self.image, cmap=plt.cm.gray)
        
        for a,i in enumerate(self.adjacency_list):
            for b in i:
                plt.plot([self.centroid_list[a][0],self.centroid_list[b][0]], [self.centroid_list[a][1],self.centroid_list[b][1]], color = 'b', linewidth = .2)
        if savefig != 0:
            plt.savefig('out.png', bbox_inches='tight', dpi = 300)
        plt.show()
        
    def visualize_graph(self, label_text = False):
        nx.draw_networkx(self.graph, node_size = 2, node_color = 'r' , width = .5, font_size = 8, with_labels = label_text)
        
    ## Call to get fit parameters for each instance
    def fit_power(self):
        from scipy.optimize import curve_fit
        components = self.get_component_masses()
        def func(x, a, b):
            return a * np.power(x,b)
            #C not necessary if zero clusters = 0----------------------------------------------------------------
            
        

        delimiters = np.logspace(0,np.log2(max(components)), int(np.log2(max(components)))+1, base = 2)
        bins = np.histogram(components, bins = delimiters)
        
        xdata = []
        ydata = []
        for i in range(len(bins[1])-1):
            xdata.append(bins[1][i+1]-bins[1][i])
        ydata = np.divide(bins[0],np.array(xdata))
    
        if len(ydata) < 2:
            return (float('nan'),float('nan'),float('nan')),float('nan'),float('nan'),float('nan'),float('nan'),float('nan'),float('nan')
        popt, pcov = curve_fit(func, xdata, ydata, p0=[500,-2], maxfev=5000)#---------------------------------

        perr = np.sqrt(np.diag(pcov))
        modelPredictions = func(xdata, *popt) 
        absError = modelPredictions - ydata
        SE = np.square(absError) # squared errors
        MSE = np.mean(SE) # mean squared errors
        RMSE = np.sqrt(MSE) # Root Mean Squared Error, RMSE
        Rsquared = 1.0 - (np.var(absError) / np.var(ydata))

        return popt, Rsquared, RMSE, pcov, perr, xdata, ydata
    
    def pickle_object(self):
        #pickle the whole object
        return 0