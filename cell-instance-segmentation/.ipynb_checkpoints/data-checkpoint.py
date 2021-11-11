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

def download_dataset(json_path, out_path, samples_per_img=100, num_threads=16, num_processes=4, selected_ids=None, crop_size=256):

    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(os.path.join(out_path, 'images'))
    os.makedirs(os.path.join(out_path, 'targets'))


    total_images = 0
    with open(json_path) as f:
        data = json.load(f)

    if selected_ids is not None:
        # Filter only selected images
        data = [img_obj for img_obj in data if img_obj['External ID'] in selected_ids]

    task_queue = Queue()
    result_queue = Queue()

    with tqdm(total=len(data)*samples_per_img) as pbar:
        for img_obj in data:

            img_url = img_obj['Labeled Data']
            if 'objects' not in img_obj['Label']:
                continue

            mask_urls = [instance['instanceURI'] for instance in img_obj['Label']['objects']]


            img = get_image_from_url(img_url)

            masks = list(ThreadPool(num_threads).imap_unordered(get_image_from_url, mask_urls))

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
    json_path = 'kidney_sample-coco-1.0/annotations/instances_default.json'
    samples_per_img = 50
    crop_size = 256
    ##########################
    print('download dataset')


    train_dataset = [
                     'image_part_001.jpg',
                     'image_part_002.jpg',
                     'image_part_003.jpg',
                     'image_part_004.jpg',
                     'image_part_005.jpg',
                     'image_part_006.jpg',
                     'image_part_007.jpg',
                     'image_part_008.jpg',

                     'MC171180.JPG',
                     'MC171177.JPG',
                     'MC171179.JPG',
                     'MC171181.JPG',
                     'MC171178.JPG',

                     '1758_part_004.jpg',
                     '1758_part_011.jpg',
                     '1758_part_008.jpg',
                     '1755_part_004.jpg',
                     '1730_part_009.jpg',
                     '0467_part_010.jpg',
                     '0467_part_003.jpg',
                     '1773_train.JPG',
                     '1133_train.JPG',
                     '0576_train.JPG',
                    ]



    download_dataset(json_path,
                     'datasets/cells_train_256',
                     samples_per_img=samples_per_img,
                     selected_ids=None,
                     crop_size=crop_size,
                     num_processes = 12,
                     num_threads = 16)

if __name__ == '__main__':
    main()
