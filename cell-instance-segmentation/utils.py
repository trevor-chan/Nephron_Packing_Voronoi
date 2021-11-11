import cv2
import matplotlib.pyplot as plt
from detectron2.data import MetadataCatalog
import random
from detectron2.utils.visualizer import Visualizer

def visualize_dataset(dataset_dicts, num_sample=3):
    MetadataCatalog.get('training_dataset').set(thing_classes=['cell'])
    metadata = MetadataCatalog.get('training_dataset')
    
    for d in random.sample(dataset_dicts, num_sample):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=2)
        vis = visualizer.draw_dataset_dict(d)
        plt.imshow(vis.get_image()[:, :, ::-1])
        plt.show()