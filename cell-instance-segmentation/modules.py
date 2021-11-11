import numpy as np
import torch
import cv2
from PIL import Image

from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.structures.instances import Instances

import data

class BrightfieldPredictor:
    def __init__(self, weights_path=None, confidence=0.7):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.DATALOADER.NUM_WORKERS = 2
        cfg.SOLVER.IMS_PER_BATCH = 2
        cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
        cfg.SOLVER.MAX_ITER = 30000
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (cell)
        cfg.INPUT.MASK_FORMAT='bitmask'
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence   # set the testing threshold for this model
        cfg.TEST.DETECTIONS_PER_IMAGE = 1000
        cfg.MODEL.DEVICE='cuda' if torch.cuda.is_available() else 'cpu'

        if weights_path is not None:
            cfg.MODEL.WEIGHTS = weights_path
        else:
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from

        self.cfg = cfg

        MetadataCatalog.get('training_dataset').set(thing_classes=['cell'])
        self.metadata = MetadataCatalog.get('training_dataset')

        self.prediction_model = DefaultPredictor(self.cfg)

    def train(self, dataset_path, max_iterations=30000):
        DatasetCatalog.register('training_dataset', lambda : data.to_coco(dataset_path))
        self.cfg.DATASETS.TRAIN = ('training_dataset',)
        self.cfg.DATASETS.TEST = ()
        self.cfg.SOLVER.MAX_ITER = max_iterations

        trainer = DefaultTrainer(self.cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()

    def predict(self, im):
            outputs = self.prediction_model(im)
            v = Visualizer(im[:, :, ::-1],
                           metadata=self.metadata,
                           scale=3.0,
                           instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
            )
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

            return v.get_image()[:, :, ::-1]

    def visualize(self, im, all_instances):
        v = Visualizer(im[:, :, ::-1],
                  metadata=self.metadata,
                  scale=2.0,
                  #instance_mode=ColorMode.IMAGE_BW
        )
        v = v.draw_instance_predictions(all_instances.to("cpu"))
        return Image.fromarray(v.get_image()[:, :, ::-1])

    def predict_large(self, im, span = 256, stride=96, nmsalg = 'poly'):
        print('run model')

        #add padding
        padding = 60
        im = np.pad(im, ((padding, padding), (padding, padding), (0, 0)),
                        mode='constant', constant_values=0)

        im_height, im_width, _ = im.shape
        all_instances = []

        for i in range(0, im_height, stride):
            for j in range(0, im_width, stride):
                sub_img = im[i:i+span, j:j+span, :]
                predictions = self.prediction_model(sub_img)
                sub_instances = predictions['instances']
                sub_instances = exclude_boundary(sub_instances, padding=60) # 30
                sub_instances = offset_instances(sub_instances, (j, i), (im_height, im_width))
                all_instances.append(sub_instances)
                all_instances.append(sub_instances)

        all_instances = Instances.cat(all_instances)
        all_instances.pred_masks = np.asarray(all_instances.pred_masks, dtype=object)

        if nmsalg == 'poly':
            all_instances = polygon_nms(all_instances)
        elif nmsalg == 'bbox':
            all_instances = bbox_nms(all_instances, overlap=0.6)
        else:
            assert False, 'nms algorithm must be polygon or bbox'

        #strip padding
        all_instances.pred_boxes.tensor -= padding
        all_instances.pred_masks = [[comp - 60 for comp in mask] for mask in all_instances.pred_masks]

        return all_instances


#def nonmax_suppression(instances):


def exclude_boundary(instances, padding):
    image_height, image_width = instances.image_size
    boxes = instances.to('cpu').pred_boxes
    """
    keep = ((boxes.tensor[:, 0] > padding) &
            (boxes.tensor[:, 1] > padding) &
            (boxes.tensor[:, 2] < image_height - padding) &
            (boxes.tensor[:, 3] < image_width - padding))
    """
    box_centers = boxes.get_centers()
    keep = ((box_centers[:, 0] > padding) &
            (box_centers[:, 1] > padding) &
            (box_centers[:, 0] < image_height - padding) &
            (box_centers[:, 1] < image_width - padding))
    return instances[keep]

    '''alright, it's time for the big boys to take over the computer typey typey
    things. nothing wrong can do to the bigger dinosaur. woW! that isn't a very
    good bOx oF cElLz.

    he attac, he defend,  but most importantly...
    he delet one char from a random spot in the program'''

def bbox_nms(instances, overlap=0.65, top_k=10000):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """
    boxes = instances.pred_boxes.tensor
    scores = instances.scores

    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]

    keep = keep[:count]
    return instances[keep.to('cpu')]

def polygon_nms(instances, score_threshold = .7, top_k=10000, nms_threshold = .5):
    from nms_altered import nms
    print('ran poly_nms')
    
    #RUN BB NMS first to preemptively cut masks - function nms is overloaded here, should rename so as not to confuse
    instances = bbox_nms(instances)

    def choose_larger(poly):
        #print(poly)
        return max(poly, key = lambda i: len(i))

    polygons = instances.pred_masks
    polygons = [choose_larger(poly) if (len(poly) > 1) else poly[0] for poly in polygons]
    polygons = [np.reshape(polygon,(int(len(np.transpose(polygon))/2),2)) for polygon in polygons]
    # pass list of lists of polygon vertices

    scores = instances.scores

    new_indices = nms(polygons, scores, score_threshold = .7, top_k = 10000, nms_threshold = .5) #, nms_algorithm=<function nms>)
    keep = torch.Tensor(new_indices).long()
    return instances[keep.to('cpu')]

def offset_boxes(boxes, offset):
    new_boxes = boxes.clone()
    i, j = offset
    for box in new_boxes:
        box[0] += i
        box[2] += i
        box[1] += j
        box[3] += j
    return new_boxes


def offset_masks(masks, offset):
    i, j = offset
    polygon_masks = []
    masks = masks.cpu()
    for mask in masks:
        polygon_mask = mask_to_polygons(mask)[0]
        #print('\n\n')
        #print(polygon_mask)
        for sub_polygon_mask in polygon_mask:
            sub_polygon_mask[::2] += i
            sub_polygon_mask[1::2] += j
        #polygon_mask[0][::2] += i
        #polygon_mask[0][1::2] += j
        polygon_masks.append(polygon_mask)
    return polygon_masks


def offset_instances(instances, offset, im_size):
    instance_dict = {
        'pred_boxes': offset_boxes(instances.pred_boxes, offset),
        'scores': instances.scores,
        'pred_classes': instances.pred_classes,
        'pred_masks': offset_masks(instances.pred_masks, offset)
    }
    return Instances(im_size, **instance_dict)

def mask_to_polygons(mask):
    # cv2.RETR_CCOMP flag retrieves all the contours and arranges them to a 2-level
    # hierarchy. External contours (boundary) of the object are placed in hierarchy-1.
    # Internal contours (holes) are placed in hierarchy-2.
    # cv2.CHAIN_APPROX_NONE flag gets vertices of polygons from contours.
    mask = np.ascontiguousarray(mask)  # some versions of cv2 does not support incontiguous arr
    res = cv2.findContours(mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    hierarchy = res[-1]
    if hierarchy is None:  # empty mask
        return [], False
    has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0
    res = res[-2]
    res = [x.flatten() for x in res]
    res = [x for x in res if len(x) >= 6]
    return res, has_holes
