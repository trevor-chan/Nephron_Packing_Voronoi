import time
import requests
import PIL
from PIL import Image
import json
from multiprocessing.pool import ThreadPool
from io import BytesIO
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import pickle
import numpy as np
import os
import shutil
from tqdm import tqdm
from multiprocessing import Process, Queue
import queue
from pycocotools.coco import COCO
import glob

import pycocotools
from detectron2.structures import BoxMode

def absolute_paths(directory):
    filenames = sorted(os.listdir(directory))
    return [os.path.join(directory, filename) for filename in filenames]

def compute_bbox(mask):
    pos = np.where(mask)
    xmin = np.min(pos[1])
    xmax = np.max(pos[1])
    ymin = np.min(pos[0])
    ymax = np.max(pos[0])
    return [xmin, ymin, xmax, ymax]

def to_coco(dataset_path):
    image_paths = [img_path for img_path in absolute_paths(os.path.join(dataset_path, 'images')) if img_path.endswith('.png')]
    #target_paths = absolute_paths(os.path.join(dataset_path, 'targets'))

    dataset_dicts = []

    for idx, image_path in enumerate(image_paths):
        target_path = os.path.join(dataset_path, 'targets', os.path.splitext(os.path.basename(image_path))[0] + '.pkl')
        with open(target_path, 'rb') as f:
            target = pickle.load(f)

        record = {}
        record['file_name'] = image_path
        record['image_id'] = idx
        record['height'] = target['size'][1]
        record['width'] = target['size'][0]

        objs = []
        for m in target['masks']:
            annotation = {'segmentation': pycocotools.mask.encode(np.asarray(m, order="F")),
                          'bbox': compute_bbox(m),
                          'bbox_mode': BoxMode.XYXY_ABS,
                          'category_id': 0,}
            objs.append(annotation)
        record['annotations'] = objs
        dataset_dicts.append(record)
    return dataset_dicts


def save_mask_target(image, masks, name, dataset_path = 'dataset'):
    image.save(os.path.join(dataset_path, 'images', name + '.png'))
    with open(os.path.join(dataset_path, 'targets', name + '.pkl'), 'wb') as f:
        pickle.dump(masks, f, protocol=pickle.HIGHEST_PROTOCOL)

def get_image_from_url(url):
    while True:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            break
        time.sleep(2)
    response.raw.decode_content = True
    img = Image.open(BytesIO(response.content)).convert('RGB')
    return img

def augment(image, masks, crop_size):
        # Brightness
        brightness_factor = np.random.normal()*0.2 + 1
        image = TF.adjust_brightness(image, brightness_factor)

        # Contrast
        contrast_factor = np.random.normal()*0.2 + 1
        image = TF.adjust_contrast(image, contrast_factor)

        #Color Jitter
        jitter = transforms.ColorJitter(hue=.1, saturation=.1)
        image = jitter(image)

        # Affine
        angle = np.random.uniform(-180, 180)
        shear = np.random.normal()*25
        scale = np.random.uniform(0.5, 2.0)
        translate = np.random.randint(-30, 30, size=2).tolist()
        image = TF.affine(image, angle, translate, scale, shear, resample=PIL.Image.BILINEAR, fillcolor=None)
        masks = [TF.affine(mask, angle, translate, scale, shear, resample=PIL.Image.BILINEAR, fillcolor=None) for mask in masks]

        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=(crop_size, crop_size))

        image = TF.crop(image, i, j, h, w)
        masks = [TF.crop(mask, i, j, h, w) for mask in masks]
        # Random horizontal flipping
        if np.random.random() > 0.5:
            image = TF.hflip(image)
            masks = [TF.hflip(mask) for mask in masks]

        # Random vertical flipping
        if np.random.random() > 0.5:
            image = TF.vflip(image)
            masks = [TF.vflip(mask) for mask in masks]

        # squeeze and binarize
        masks = [(np.array(mask)[:, :, 0] > 0.5).astype(np.uint8) for mask in masks]

        # prune masks that have no object or only a sliver of an object
        masks = [mask for mask in masks if mask[10:-10, 10:-10].any()]
        return image, masks

class Worker(Process):
    def __init__(self, task_queue, result_queue, img, masks, out_path, crop_size):
        super().__init__()
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.img = img
        self.masks = masks
        self.out_path = out_path
        self.crop_size = crop_size

    def run(self):
        proc_name = self.name
        while True:#not stopping.is_set():
            try:
                index = self.task_queue.get(True, 1)
                sub_img, sub_masks = augment(self.img, self.masks, self.crop_size)
                target = {'masks': sub_masks, 'size': sub_img.size}
                save_mask_target(sub_img, target, f'{index:05d}', dataset_path=self.out_path)
                self.result_queue.put(index)
            except queue.Empty:
                return

def build_dataset(json_path, img_path, out_path, samples_per_img=100, num_threads=16, num_processes=4, selected_ids=None, crop_size=256):

    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(os.path.join(out_path, 'images'))
    os.makedirs(os.path.join(out_path, 'targets'))
    
    coco = COCO(json_path)
    total_images = len(coco.imgs)
    # total_images = 0
    # with open(json_path) as f:
    #     data = json.load(f)
    
    # Filter only selected images
    if selected_ids is not None:
        data = [img_obj for img_obj in data if img_obj['External ID'] in selected_ids]
    
    
    task_queue = Queue()
    result_queue = Queue()

    with tqdm(total=total_images*samples_per_img) as pbar:
        for i in coco.imgs:
            img_obj = coco.imgs[i]
            
            cat_ids = coco.getCatIds()
            anns_ids = coco.getAnnIds(imgIds=img_obj['id'], catIds=cat_ids, iscrowd=None)
            anns = coco.loadAnns(anns_ids)

            img = Image.open(os.path.join(img_path, img_obj['file_name'])).convert('RGB')
            masks = [coco.annToMask(ann) for ann in anns]
            masks = [Image.fromarray(np.uint8(mask)*255).convert('RGB') for mask in masks]

            for _ in range(samples_per_img):
                task_queue.put(total_images)
                total_images += 1

            workers = []
            for proc_index in range(num_processes):
                p = Worker(task_queue, result_queue, img, masks, out_path, crop_size)
                p.daemon = True
                p.start()
                workers.append(p)

            for worker in workers:
                worker.join(200)
            for worker in workers:
                if worker.is_alive():
                    print("Process timed out")

            pbar.update(samples_per_img)
            # for index in range(samples_per_img):
            # while True:
            #     try:
            #         i = result_queue.get(True, 10)
            #         pbar.update(1)
            #     except queue.Empty:
            #         break
            



def main():
    ##########################
    dataset_path = 'datasets/kidney_actual'
    json_path = glob.glob('{}/annotations/*'.format(dataset_path))[0]
    img_path = '{}/images/'.format(dataset_path)
    samples_per_img = 200
    crop_size = 512
    ##########################
    print('building dataset')


    train_dataset = [
                        'MAX_E14_1.png',
                        'MAX_E14_2.png',
                        'MAX_E15_1.png',
                        'MAX_E15_2.png',
                        'MAX_E15_3.png',
                        'MAX_E16_1.png',
                        'MAX_E17_1.png',
                        'MAX_E18_1.png',
                        'MAX_E18_3.png',
                    ]

    build_dataset(json_path, img_path,
                     'datasets/kidney_training',
                     samples_per_img=samples_per_img,
                     selected_ids=None,
                     crop_size=crop_size,
                     num_processes = 48,
                     num_threads = 16)

if __name__ == '__main__':
    main()
