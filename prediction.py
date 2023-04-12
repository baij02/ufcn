import utils
import logging
import os
import sys
from torch.utils.data import DataLoader
from glob import glob

import torch
from PIL import Image
import argparse
import monai
from monai.data import PersistentDataset, list_data_collate, SmartCacheDataset, partition_dataset
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai import transforms as mt
from monai.visualize import plot_2d_or_3d_image
import random
import ufcn
import load_FFSData
import load_FFSDataTest
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
import imageio

pjoin = os.path.join

def get_transforms():
    train_trans = mt.Compose(
        [
            mt.LoadImageD(keys=['img', 'seg']),
            mt.EnsureChannelFirstd(keys=['img', 'seg']),
            mt.ScaleIntensityD(keys=['img',"seg"]),
            mt.ToTensorD(keys=['img', 'seg']),
        ]
    )

    val_trans = mt.Compose(
        [
            mt.LoadImageD(keys=['img', 'seg']),
            mt.EnsureChannelFirstd(keys=["img", "seg"]),
            mt.ScaleIntensityD(keys=['img',"seg"]),
            mt.ToTensorD(keys=['img', 'seg']),
        ]
    )
    return train_trans, val_trans

def main(args):

    testDataset = load_FFSDataTest.CustomDataset()
    
    testDataload = DataLoader(testDataset, batch_size=args.batchsize, shuffle=True)
    print("number batch of test",testDataload.__len__())
    print("data load done")


    dice_metric = DiceMetric(include_background=True, reduction="mean")
    post_trans = mt.Compose([
        mt.Activations(sigmoid=True)
    ])

    
    # create UNet, DiceLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ufcn.UFCN(activation = args.activation, threshold = args.thresh)

    model = torch.nn.DataParallel(model)

    model.load_state_dict( torch.load("checkpoints/" + args.modelWeight))
    # model = torch.load("checkpoints/ufcn_e100l0d1Relu.pth")
    # #model.apply(utils.init_weights)

    # model = model.load_state_dict()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)
    model.to(device)

    print("model loaded")
    loss_function = monai.losses.ssim_loss.SSIMLoss(spatial_dims=2)
    bce = torch.nn.BCELoss()
    # optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=1e-5)

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max')
    # start a typical PyTorch training
    val_interval = 1
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()

    outputPreq = 0
 
   
    model.eval()
    f = open('silu.txt', 'w')
    with torch.no_grad():
        metric_sum = 0.0
        metric_count = 0
        val_images = None
        val_labels = None
        val_outputs = None
        for inputs1, inputs2, label, seg in testDataload:
            inputs1 = inputs1.to(device)
            inputs2 = inputs2.to(device)
            label = label.to(device)
            seg = seg.to(device)

            val_outputs, val_seg, labelLoss = model(inputs1, inputs2)
            val_seg1 = torch.squeeze(val_seg)
            output = val_seg1.cpu().detach().numpy()
            print(testDataset.fileName)
            layerLoss = 0
            term = 1
            prediction = (labelLoss[0].cpu().detach().numpy()[0][0] + labelLoss[1].cpu().detach().numpy()[0][0] + labelLoss[2].cpu().detach().numpy()[0][0])/3

        metric = metric_sum / metric_count
        metric_values.append(metric)
    
    print("Test Dice is: ", metric)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--data", default='gd', type=str)
    parser.add_argument("--arch", default='ufcn', type=str)
    parser.add_argument("--val_inter", default=1, type=int)
    parser.add_argument("--batchsize", default=1, type=int)
    parser.add_argument("--test_batch_size", default=1, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--fast", default=False, type=bool)
    parser.add_argument("--dataDic", default='./train_data')
    parser.add_argument("--lr", default=1e-2, type=float)
    parser.add_argument("--ext", default='unet', type=str)
    parser.add_argument("--pref", default=20, type=int)
    parser.add_argument("--activation", default='relu', type=str)
    parser.add_argument("--thresh", default=0.02, type=float)
    parser.add_argument("--modelWeight", default=0.02, type=float)
    args = parser.parse_args()
    print(args)
    
    main(args)