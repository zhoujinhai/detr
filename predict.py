# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse 
import random 
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
import util.misc as utils 
from engine import evaluate, train_one_epoch, predict
from models import build_model
import matplotlib.pyplot as plt 
from PIL import Image


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--tooth_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--strict', action="store_false", help="set strict when load_state_dict")
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # predict
    parser.add_argument('--img_path', default=None, type=str, help='the path of predict image')
    parser.add_argument('--img_dirs', default=None, type=str, help='the dir of predict images')
    return parser


# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def detect(im, model, transform, threshold=0.7):
    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)
    print("image.shape:", img.shape)
    # demo model only support by default images with aspect ratio between 0.5 and 2
    # if you want to use images with an aspect ratio outside this range
    # rescale your image so that the maximum size is at most 1333 for best results
    # assert img.shape[-2] <= 1600 and img.shape[-1] <= 1600, 'demo model only supports images up to 1600 pixels on each side'

    # propagate through the model
    outputs = model(img)

    # keep only predictions with 0.7+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > threshold

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
    return probas[keep], bboxes_scaled

def to_tuple(tup):
            if isinstance(tup, tuple):
                return tup
            return tuple(tup.cpu().tolist())


def id2rgb(id_map):
    if isinstance(id_map, np.ndarray):
        id_map_copy = id_map.copy()
        rgb_shape = tuple(list(id_map.shape) + [3])
        rgb_map = np.zeros(rgb_shape, dtype=np.uint8)
        for i in range(3):
            rgb_map[..., i] = id_map_copy # % 256
            # id_map_copy //= 256
        return rgb_map
    color = []
    for _ in range(3):
        color.append(id_map % 256)
        id_map //= 256
    return color
    # This helper function creates the final panoptic segmentation image
    # It also returns the area of the masks that appears on the image

    m_id = masks.transpose(0, 1).softmax(-1)

    if m_id.shape[-1] == 0:
        # We didn't detect any mask :(
        m_id = torch.zeros((h, w), dtype=torch.long, device=m_id.device)
    else:
        m_id = m_id.argmax(-1).view(h, w)

    if dedup:
        # Merge the masks corresponding to the same stuff class
        for equiv in stuff_equiv_classes.values():
            if len(equiv) > 1:
                for eq_id in equiv:
                    m_id.masked_fill_(m_id.eq(eq_id), equiv[0])

    final_h, final_w = to_tuple(target_size)

    seg_img = Image.fromarray(id2rgb(m_id.view(h, w).cpu().numpy()))
    seg_img = seg_img.resize(size=(final_w, final_h), resample=Image.NEAREST)

    np_seg_img = (
        torch.ByteTensor(torch.ByteStorage.from_buffer(seg_img.tobytes())).view(final_h, final_w, 3).numpy()
    )
    m_id = torch.from_numpy(rgb2id(np_seg_img))

    area = []
    for i in range(len(scores)):
        area.append(m_id.eq(i).sum().item())
    return area, seg_img


def rgb2id(color):
    if isinstance(color, np.ndarray) and len(color.shape) == 3:
        if color.dtype == np.uint8:
            color = color.astype(np.int32)
        return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
    return int(color[0] + 256 * color[1] + 256 * 256 * color[2])


def detectSeg(im, model, transform, threshold=0.7):
    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)
    print("image.shape:", img.shape, img.size, im.size) 
    # propagate through the model
    outputs = model(img)
    # print("outputs: ", outputs)
    # keep only predictions with 0.7+ confidence  
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]  # -1是no object 
    keep = probas.max(-1).values > threshold 
    scores, cur_classes = outputs['pred_logits'].softmax(-1).max(-1) 
    # print("scores: ", scores, scores.shape, " class: ", cur_classes, cur_classes.shape)
    # keep1 = cur_classes[0].ne(outputs["pred_logits"].shape[-1] - 1) & (scores > threshold)  
    # print("keep:", keep, "keep1: ", keep1)
    cur_classes = cur_classes[0, keep] 
    print("cur_classes: ", cur_classes, cur_classes.shape)
    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
    print("bboxes_scaled: ", bboxes_scaled, bboxes_scaled.shape)
    masks = outputs["pred_masks"][0, keep]
    # print("masks: ", masks.shape)
    masks = interpolate(masks[:, None], (im.size[1], im.size[0]), mode="bilinear").squeeze(1)  
    # print("masks shape: ", masks.shape)

    outputs_masks = (masks.sigmoid() > 0.5) * 255

    # # Remove pixel that out of box
    masks_scaled = np.zeros_like(outputs_masks.detach().numpy()) 
    print("--------------", masks_scaled.shape)
    for idx in range(masks.shape[0]):
        cur_mask = outputs_masks[idx].detach().numpy()
        cur_box = bboxes_scaled[idx]
        rows_to_keep = slice(int(cur_box[1]), int(cur_box[3]))  # 指定行
        cols_to_keep = slice(int(cur_box[0]), int(cur_box[2]))  # 指定列

        # 创建一个与cur_mask大小相同的全0矩阵
        zero_matrix = np.zeros_like(cur_mask) 
        zero_matrix[rows_to_keep, cols_to_keep] = cur_mask[rows_to_keep, cols_to_keep]
        masks_scaled[idx] = zero_matrix

     # for idx in range(masks_scaled.shape[0]):
     #     cur_mask = masks_scaled[idx] 
     #     seg_img = Image.fromarray(id2rgb(cur_mask)) 
     #     seg_img.save("./output/seg299/test" + str(idx) + ".png") 
    
     return probas[keep], bboxes_scaled, masks_scaled
 

def plot_results(pil_img, prob, boxes, output):
    CLASSES = [
        'N/A', 'teeth'
    ]

    # colors for visualization
    COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
            [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), COLORS * 100):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.savefig(output)
    # plt.show()


def plot_seg_results(pil_img, prob, boxes, masks, output):
    # CLASSES = [
    #     'N/A', 'teeth'
    # ]
    CLASSES = [
        'N/A', 'L8', 'L7', 'L6', 'L5', 'L4', 'L3', 'L2', 'L1',
        'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8'
        # 'N/A', 'tooth', 'tooth', 'tooth', 'tooth', 'tooth', 'tooth', 'tooth', 'tooth',
        # 'tooth', 'tooth', 'tooth', 'tooth', 'tooth', 'tooth', 'tooth', 'tooth'
    ]

    # colors for visualization
    COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
            [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]] 
    COLORS = COLORS * len(masks)
     
    rgb_shape = tuple(list(masks.shape[1:]) + [3])
    rgb_map = np.zeros(rgb_shape)
    for idx, mask in enumerate(masks):
        id_map = mask.copy()
        for i in range(3):
            rgb_map[..., i] += id_map * COLORS[idx][i]  
    
    img_masks = Image.fromarray(rgb_map.astype(np.uint8)) 
    # img_masks.save("./output/seg299/masks.png")
    # merge mask 
     
    pil_img = Image.blend(img_masks, pil_img, alpha=0.5) 
    # pil_img.save("./output/seg299/merge.png") 
      
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), COLORS):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.savefig(output)
    plt.close()
    # plt.show()


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    output_dir = Path(args.output_dir)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu') 
        model.load_state_dict(checkpoint['model'], strict=args.strict)
        print("load model {} is success!".format(args.resume))
    else:
        print("Don't load model!")
        return
    
    # standard PyTorch mean-std input image normalization
    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    if args.img_path is not None:
        assert Path(args.img_path).is_file(), "{} not an image path".format(args.img_path)
        im = Image.open(img_path)
        scores, boxes = detect(im, model, transform=transform)
        print("scores: ", scores)
        print("boxes: ", boxes)

    if args.img_dirs is not None:
        assert Path(args.img_dirs).is_dir(), "{} not a dir path".format(args.img_dirs)
        img_paths = Path(args.img_dirs).glob("*.jpg")
        # print("loads {} images".format(len(list(img_paths))))
        for idx, img_path in enumerate(img_paths):
            print(img_path)
            im = Image.open(img_path)
            masks = None
            if args.masks:
                scores, boxes, masks = detectSeg(im, model, transform=transform)
            else: 
                scores, boxes = detect(im, model, transform=transform)
            # print(" scores: ", scores)
            # print("boxes: ", boxes)
            out_path = Path(output_dir) / img_path.name
            print("out_path: ", out_path)
            if args.masks:
                plot_seg_results(im, scores, boxes, masks, out_path)
            else:
                plot_results(im, scores, boxes, out_path)
        img_paths = Path(args.img_dirs).glob("*.png")
        # print("loads {} images".format(len(list(img_paths))))
        for idx, img_path in enumerate(img_paths):
            print(img_path)
            im = Image.open(img_path)
            print("im.shape", im.size)
            if im.mode == 'RGBA':
                im = im.convert('RGB')
            masks = None
            if args.masks:
                scores, boxes, masks = detectSeg(im, model, transform=transform)
            else: 
                scores, boxes = detect(im, model, transform=transform)
            # print(" scores: ", scores)
            # print("boxes: ", boxes)
            out_path = Path(output_dir) / img_path.name
            print("out_path: ", out_path)
            if args.masks:
                plot_seg_results(im, scores, boxes, masks, out_path)
            else:
                plot_results(im, scores, boxes, out_path)
    # print("results: ", results)
    return

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print("args: ", args)
    main(args)
