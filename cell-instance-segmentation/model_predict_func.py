from modules import BrightfieldPredictor
import cv2
import os
import matplotlib.pyplot as plt
import pickle
import sys
from sys import argv
from detectron2.structures import Instances, Boxes

model = BrightfieldPredictor(weights_path='./models/bright-field.pth', confidence=0.6)

assert len(argv) > 1, "missing data file"

file_name = argv[1]
rerun = 0
if len(argv) > 2:
    if argv[2] == 'rerun':
        rerun = 1

#if both output files exist, exit
if os.path.isfile(file_name[0:-4]+'_instances.data') and os.path.isfile(file_name[0:-4]+'_visual.JPG') and rerun == 0:
    sys.exit(file_name[0:-4]+'_instances.data'+' and '+file_name[0:-4]+'_visual.JPG'+' already exist')

image = cv2.imread(file_name)

#if instance file does not exist, run model; else, load in instance file to Instances object (required for visualizer)
if not os.path.isfile(file_name[0:-4]+'_instances.data') or rerun == 1:
    
    instances = model.predict_large(image)
    #instances = instances.to('cpu')
    instance_dict = {
        "pred_boxes":instances.pred_boxes.tensor,
        "pred_masks":instances.pred_masks,
        "scores":instances.scores,
        "classes":instances.pred_classes,
        "image_size": instances.image_size,
    }
    with open(file_name[0:-4]+'_instances.data', 'wb') as filehandle:
        pickle.dump(instance_dict, filehandle)
        
else:
    #load an instances object from reading file
    print('load instances from '+file_name[0:-4]+'_instances.data')
    with open(file_name[0:-4]+'_instances.data', 'rb') as filehandle:
        instance_dict = pickle.load(filehandle)
    
    kwargs = {
            'pred_boxes':Boxes(instance_dict['pred_boxes']),
            'pred_masks':instance_dict['pred_masks'],
            'scores':instance_dict['scores'],
            'pred_classes':instance_dict['classes'],
            }
    instances = Instances(instance_dict['image_size'], **kwargs)
    
#if check here not really necessary, visual output will never be produced unless the model has been run
if not os.path.isfile(file_name[0:-4]+'_visual.JPG') or rerun == 0:
    print('output verification saved to '+file_name[0:-4]+'_visual.JPG')
    out_img = model.visualize(image,instances)

    out_img.save(file_name[0:-4]+'_visual.JPG')
