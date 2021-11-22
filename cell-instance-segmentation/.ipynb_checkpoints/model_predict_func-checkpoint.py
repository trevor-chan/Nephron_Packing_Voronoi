from modules import BrightfieldPredictor
import cv2
import os
import matplotlib.pyplot as plt
import pickle
import sys
from sys import argv
from detectron2.structures import Instances, Boxes

    
def main(input_file, output_dir = None, rerun = False):
    model = BrightfieldPredictor(weights_path='./models/256/model_final.pth', confidence=0.6)
    
    if not sys.warnoptions:
        import warnings
        warnings.simplefilter("ignore")

    file_name = input_file
    if output_dir is not None:
        output_dir = "{}/{}".format(output_dir, os.path.splitext(os.path.basename(file_name))[0])
    else:
        output_dir = os.path.splitext(file_name)[0]
    
    #if both output files exist, exit
    if os.path.isfile(output_dir+'_instances.data') and os.path.isfile(output_dir+'_visual.JPG') and rerun == False:
        sys.exit(output_dir+'_instances.data'+' and '+output_dir+'_visual.JPG'+' already exist')

    image = cv2.imread(file_name)

    #if instance file does not exist, run model; else, load in instance file to Instances object (required for visualizer)
    if not os.path.isfile(output_dir+'_instances.data') or rerun == True:

        instances = model.predict_large(image)
        instances = instances.to('cpu')
        instance_dict = {
            "pred_boxes":instances.pred_boxes.tensor,
            "pred_masks":instances.pred_masks,
            "scores":instances.scores,
            "classes":instances.pred_classes,
            "image_size": instances.image_size,
        }
        with open(output_dir+'_instances.data', 'wb') as filehandle:
            pickle.dump(instance_dict, filehandle)

    else:
        #load an instances object from reading file
        print('load instances from '+output_dir+'_instances.data')
        with open(output_dir+'_instances.data', 'rb') as filehandle:
            instance_dict = pickle.load(filehandle)

        kwargs = {
                'pred_boxes':Boxes(instance_dict['pred_boxes']),
                'pred_masks':instance_dict['pred_masks'],
                'scores':instance_dict['scores'],
                'pred_classes':instance_dict['classes'],
                }
        instances = Instances(instance_dict['image_size'], **kwargs)


    #if check here not really necessary, visual output will never be produced unless the model has been run
    if not os.path.isfile(output_dir+'_visual.JPG') or rerun == True:
        print('output verification saved to '+output_dir+'_visual.JPG')
        out_img = model.visualize(image,instances)

        out_img.save(output_dir+'_visual.JPG')

if __name__ == '__main__':
    assert len(argv) > 1, "missing data file"
    inputfile = argv[1]
    if len(argv) > 2:
        output_directory = argv[2]
    rerun = 0
    if len(argv) > 3:
        rerun = 1
    main(inputfile, output_directory, rerun)