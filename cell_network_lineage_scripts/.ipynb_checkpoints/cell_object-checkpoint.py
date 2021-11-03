import sys
import math
#import matplotlib.pyplot as plt
import numpy as np
import glob
#import os
from PIL import Image, ImageDraw
#import torch

#from tqdm import tqdm

#import pickle
import cv2

#from skimage.draw import ellipse
from skimage.measure import label, regionprops, find_contours, approximate_polygon
from skimage.transform import rotate
#from sklearn.neighbors import KDTree
#from scipy.spatial import distance

class cell_object:
    #Cell object - takes in a segmentation output and initializes a cell-object with notable properties calculated
    #
    
    def __init__(self, pred_box, polygon, score):
        self.boundingbox = pred_box
        
        scale_factor = 0.943  #(for distance, for area = .889249) - determined by microscope scale: micron to pixel
        
        if len(polygon) == 1:
            self.polygon = polygon[0]
        else:
            self.polygon = max(polygon, key = len)
        self.score = score
        
        self.mask = self.construct_b_mask(self.polygon,self.boundingbox)
        self.perimeter = self.calc_perimeter(self.polygon) * scale_factor
        
        self.centroid = self.calc_centroid(self.boundingbox)
        
        region_props = self.cell_poly_properties(self.mask)
        
        self.area = region_props[0] * scale_factor * scale_factor
        self.orientation = region_props[1]
        self.majaxis = region_props[2] * scale_factor
        self.minaxis = region_props[3] * scale_factor
        self.circularity = self.calc_circularity(self.area,self.perimeter)
        
    def construct_b_mask(self, polygon, bbox):
        buffer = 5
        width = int(abs(bbox[0]-bbox[2]))+2*buffer
        height = int(abs(bbox[1]-bbox[3]))+2*buffer
        result = np.copy(polygon)
        result[0::2] = np.subtract(result[0::2], min(result[0::2])-buffer)
        result[1::2] = np.subtract(result[1::2], min(result[1::2])-buffer)
        
        img = Image.new('1', (width, height), 0)
        img1 = ImageDraw.Draw(img)
        img1.polygon(result.tolist(), fill = 1, outline = 1)
        return np.array(img)
    
    def calc_centroid(self, bbox):
        return (((bbox[0]+bbox[2])/2).item() , ((bbox[1]+bbox[3])/2).item())
        
    def calc_perimeter(self, polygon):
        perimeter = 0
        for i in range (0,len(polygon)-4,2):
            perimeter += math.sqrt((polygon[i]-polygon[i+2])**2 + (polygon[i+1]-polygon[i+3])**2)
        perimeter += math.sqrt((polygon[0]-polygon[len(polygon)-2])**2 + (polygon[1]-polygon[len(polygon)-1])**2)
        return perimeter
    
    def cell_poly_properties(self, img):
        #takes in an array of a binary mask
        #finds notable cell properties
        #returns a cell array of user defined objects

        img = label(img)
        regions = regionprops(img)
        #print(regions[0].perimeter)
        
        cell_properties = (regions[0].area, regions[0].orientation, regions[0].major_axis_length, 
                           regions[0].minor_axis_length)

        return cell_properties
    
    def calc_circularity(self, area, perimeter):
        return (area * 4 * math.pi / (perimeter**2))
    
    def pickle_object(self):
        #pickle whole object
        return 0
    