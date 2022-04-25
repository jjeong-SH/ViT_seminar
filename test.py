import os
import cv2
import torch
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import ImageNetDataset, MNISTDataset, CIFARDataset

if __name__ == '__main__':
    path = 'data'
    mnist_train, mnist_val = MNISTDataset(path, 32, val_ratio=0.2)
    mnist_dataloader = DataLoader(mnist_train, batch_size=32, shuffle=True)
    kjafakjlsdjf
    print(len(mnist_train), len(mnist_val))

    cifar_train, cifar_val = CIFARDataset(path, 32, val_ratio=0.2)
    cifar_dataloader = DataLoader(cifar_train, batch_size=32, shuffle=True)
    print(len(cifar_train), len(cifar_val))

    imgnet = ImageNetDataset(path, 32, val_ratio=0.2)
    img_train, img_val = imgnet.split_dataset()
    imgnet_dataloader = DataLoader(img_train, batch_size=32, shuffle=True, drop_last=True)
    print(len(img_train), len(img_val))
    print("----------------")

    img, label = next(iter(imgnet_dataloader))
    print("이미지 크기 :{}".format(img.shape))
    print("레이블 : {}".format(label))
    plt.imshow(torchvision.utils.make_grid(img, normalize=False, nrow=4).permute(1, 2, 0))
    plt.show()
