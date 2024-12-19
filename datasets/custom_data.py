# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset 
from panopticapi.utils import rgb2id
from util.box_ops import masks_to_boxes

from .coco import make_coco_transforms

from pycocotools import mask as maskUtils
import random


class COCOAnnotationTransform(object):
    """Transforms a COCO annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    """
    def __init__(self):
        pass

    def __call__(self, target):
        """
        Args:
            target (dict): COCO target json annotation as a python dict 
        Returns:
            a list containing lists of bounding boxes  [bbox coords]
        """ 
        res = []
        for obj in target:
            if 'bbox' in obj:
                bbox = obj['bbox']   
                final_box = list(np.array([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]])) 
                res += [final_box]  # [xmin, ymin, xmax, ymax]
            else:
                print("No bbox found for object ", obj)

        return res


class Tooth(Dataset):
    def __init__(self, img_folder, ann_file, transforms=None, return_masks=True):
        from pycocotools.coco import COCO
        self.coco = COCO(ann_file)
        # sort 'images' field so that they are aligned with 'annotations'
        # i.e., in alphabetical order 
        self.ids = list(self.coco.imgToAnns.keys())
        if len(self.ids) == 0:
            self.ids = list(self.coco.imgs.keys()) 
        self.cates = self.coco.getCatIds()
        # print("cates: ", self.cates, "ids: ", len(self.ids))

        self.img_folder = img_folder 
        self.ann_file = ann_file
        
        self.coco_tf = COCOAnnotationTransform()
        self.transforms = transforms
        self.return_masks = return_masks
        # print("retrun_mask: ", return_masks)

        # img_id = self.ids[0] 
        # ann_ids = self.coco.getAnnIds(imgIds=img_id) 
        # print("000000000000", ann_ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        # print("img_id: ", img_id)
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids) 
        anns = [x for x in anns if x['image_id'] == img_id]
        # print("anns: ", anns)

        # Separate out crowd annotations. These are annotations that signify a large crowd of
        # objects of said class, where there is no annotation for each individual object. Both
        # during testing and training, consider these crowds as neutral.
        crowd  = [x for x in anns if     ('iscrowd' in x and x['iscrowd'])]
        anns = [x for x in anns if not ('iscrowd' in x and x['iscrowd'])]
        num_crowds = len(crowd)
        for x in crowd:
            x['category_id'] = -1

        # This is so we ensure that all crowd annotations are at the end of the array
        anns += crowd

        # The split here is to have compatibility with both COCO2014 and 2017 annotations.
        # In 2014, images have the pattern COCO_{train/val}2014_%012d.jpg, while in 2017 it's %012d.jpg.
        # Our script downloads the images as %012d.jpg so convert accordingly.
        file_name = self.coco.loadImgs(img_id)[0]['file_name']
        # print("********", file_name, file_name.split("\\"))
        if file_name.startswith('COCO'):
            file_name = file_name.split('_')[-1]
        if file_name.find("\\") != -1:
            file_name = "/".join(file_name.split("\\"))
 
        img_path = Path(self.img_folder) / file_name 
        assert Path(img_path).exists, "Image path does not exist: {}".format(img_path)

        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        assert h == self.get_height_and_width(idx)[0] and w == self.get_height_and_width(idx)[1]

        if len(anns) > 0:
            # Pool all the masks for this image into one [num_objects,height,width] matrix
            masks = [self.coco.annToMask(obj).reshape(-1) for obj in anns]
            masks = np.vstack(masks)
            masks = masks.reshape(-1, h, w)
            masks = torch.as_tensor(masks, dtype=torch.uint8)
            labels = torch.tensor([ann['category_id'] for ann in anns], dtype=torch.int64)

        target = {}
        target['image_id'] = torch.tensor(img_id)
        if self.return_masks:
            target['masks'] = masks
            target["boxes"] = masks_to_boxes(masks)
            # print("box.shape", type(target['boxes']), target['boxes'].size())
        else:
            target["boxes"] = torch.tensor(self.coco_tf(anns), dtype=torch.float32)
            # print("box.shape", type(target['boxes']), target['boxes'].size())
        target['labels'] = labels 

        target['size'] = torch.as_tensor([int(h), int(w)])
        target['orig_size'] = torch.as_tensor([int(h), int(w)])
        if "segmentation" in anns[0]:
            for name in ['iscrowd', 'area']:
                target[name] = torch.tensor([ann[name] for ann in anns])
        # print("target: ", target, len(anns), num_crowds)
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        return img, target

    def __len__(self):
        return len(self.ids)

    def get_height_and_width(self, idx):
        img_info = self.coco.loadImgs(self.ids[idx])[0]
        # print("img_info: ", img_info)
        height = img_info['height']
        width = img_info['width']
        return height, width


def build(image_set, args):
    img_folder_root = Path(args.tooth_path) 
    assert img_folder_root.exists(), f'provided tooth path {img_folder_root} does not exist'  
    PATHS = {
        "train": (Path('train'), Path('train') / 'annotations.json'),
        "val": (Path('val'), Path('val') / 'annotations.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    img_folder_path = img_folder_root / img_folder 
    ann_file = img_folder_root / ann_file
    print("**************", img_folder_path, ann_file, image_set)
    dataset = Tooth(img_folder_path, ann_file,
                           transforms=make_coco_transforms(image_set), return_masks=args.masks)

    return dataset
      
