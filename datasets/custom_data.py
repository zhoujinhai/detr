# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved 
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset  
from util.box_ops import masks_to_boxes
 
import torchvision
from pycocotools import mask as coco_mask 
import datasets.transforms as T


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
        print("loading Data: ", len(self.ids))

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


class Tooth1(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks):
        super(Tooth1, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        img, target = super(Tooth1, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


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
    print("******loading {} Data********".format(image_set), img_folder_path, ann_file, image_set)
    dataset = Tooth(img_folder_path, ann_file,
                           transforms=make_coco_transforms(image_set), return_masks=args.masks)

    return dataset


def build1(image_set, args):
    img_folder_root = Path(args.tooth_path) 
    assert img_folder_root.exists(), f'provided tooth path {img_folder_root} does not exist'  
    PATHS = {
        "train": (Path('train'), Path('train') / 'annotations.json'),
        "val": (Path('val'), Path('val') / 'annotations.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    img_folder_path = img_folder_root / img_folder 
    ann_file = img_folder_root / ann_file
    print("******loading {} Data********".format(image_set), img_folder_path, ann_file, image_set)
    dataset = Tooth1(img_folder_path, ann_file,
                           transforms=make_coco_transforms(image_set), return_masks=args.masks)

    return dataset


if __name__ == "__main__": 
    IMG_DIR = r"/home/jinhai_zhou/data/2D_seg/val"
    JSON_PATH = r"/home/jinhai_zhou/data/2D_seg/val/annotations.json"

    dataset = Tooth(IMG_DIR, JSON_PATH)
    data = dataset[0]

    
        

      
