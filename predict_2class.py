from torch.utils.data import dataset
from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np

from torch.utils import data
from datasets import VOCSegmentation, Cityscapes, cityscapes
from torchvision import transforms as T
from metrics import StreamSegMetrics

import torch
import torch.nn as nn

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from glob import glob

import cv2
import imageio
import skimage.io
import pylab

def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--input", type=str, default='./test',
                        # required=True,
                        help="path to a single image or image directory")
    parser.add_argument("--dataset", type=str, default='cityscapes',
                        choices=['voc', 'cityscapes'], help='Name of training set')

    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )

    parser.add_argument("--model", type=str, default='deeplabv3plus_resnet50',
                        choices=available_models, help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=True,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--save_val_results_to", default='./test',
                        help="save segmentation results to the specified dir")

    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--val_batch_size", type=int, default=1,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)

    
    parser.add_argument("--ckpt", default='./checkpoints/best/best_deeplabv3plus_resnet50_UAS_os16.pth', type=str,
                        help="resume from checkpoint")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    return parser

def main():
    opts = get_argparser().parse_args()
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
        decode_fn = VOCSegmentation.decode_target
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 2
        decode_fn = Cityscapes.decode_target
    elif opts.dataset.lower() == 'uas':
        opts.num_classes = 2
        decode_fn = Cityscapes.decode_target

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup dataloader
    image_files = []
    if os.path.isdir(opts.input):
        for ext in ['png', 'jpeg', 'jpg', 'JPEG']:
            files = glob(os.path.join(opts.input, '**/*.%s'%(ext)), recursive=True)
            if len(files)>0:
                image_files.extend(files)
    elif os.path.isfile(opts.input):
        image_files.append(opts.input)
    
    # Set up model (all models are 'constructed at network.modeling)
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        print("Resume model from %s" % opts.ckpt)
        del checkpoint
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    #denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    if opts.crop_val:
        transform = T.Compose([
                T.Resize(opts.crop_size),
                T.CenterCrop(opts.crop_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
    else:
        transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
    if opts.save_val_results_to is not None:
        os.makedirs(opts.save_val_results_to, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    cap = cv2.VideoCapture('37b49c1476b7cf8b9f178ff65da433fc.mp4')
    fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))
    # pred_img = np.zeros((height, width, 3)).astype(np.uint8)

    with torch.no_grad():
        model = model.eval()

        while(cap.isOpened()):
            ret, frame = cap.read()
            # cv2.imshow('input', frame)
            if (ret):
                img_ori = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img2 = img_ori.copy()
                # img2 = frame.copy()
                # for img_path in tqdm(image_files):
                #     ext = os.path.basename(img_path).split('.')[-1]
                #     img_name = os.path.basename(img_path)[:-len(ext)-1]
                #     img = Image.open(img_path).convert('RGB')
                img = transform(img_ori).unsqueeze(0) # To tensor of NCHW
                # img = torch.from_numpy(img_ori).unsqueeze(0)
                # img2 = img.squeeze(0).permute(1, 2, 0).numpy().copy()
                img = img.to(device)

                pred_mask = model(img).max(1)[1].cpu().numpy()[0] # HW
                # colorized_preds = decode_fn(pred).astype('uint8')
                binary = np.uint8(pred_mask * 255)
                # gray = cv2.cvtColor(pred_mask, cv2.COLOR_BGR2GRAY)
                # cv2.imshow('input', binary)
                # ret, binary = cv2.threshold(np.uint8(pred_mask), 1, 255, 0)

                contour, h = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

                for i in range(len(contour)):
                    if cv2.contourArea(contour[i]) > 8000:
                        approx = cv2.approxPolyDP(contour[i], 20, True)
                        img2 = cv2.drawContours(img2, [approx], 0, (255, 255, 255), 5)
                out.write(img2)
                # for i in range(3):
                #     pred_img[:,:,i] = (img_ori[:,:,i] * pred_mask)




            else:
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # colorized_preds = Image.fromarray(colorized_preds)
        # if opts.save_val_results_to:
        #     colorized_preds.save(os.path.join(opts.save_val_results_to, img_name+'_label.png'))

if __name__ == '__main__':
    main()
