import argparse
import os
import json
import random
import time
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
from importlib import import_module
from dataset import MNISTDataset, CIFARDataset, ImageNetDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from resnet_pytorch.resnet import ResNet50


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True


def train(args):
    seed_everything(args.seed)  # def: 123

    # -- settings
    use_cuda = torch.cuda.is_available()
    print("USE CUDA> ", use_cuda)
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    dataset_module = getattr(import_module("dataset"), args.dataset)
    if args.dataset in ['MNISTDataset', 'CIFARDataset']:
        num_classes = 10
        train_set, val_set = dataset_module(path='data', resize=args.resize, val_ratio=args.val_ratio)
    elif args.dataset == 'ImageNetDataset':
        num_classes = 20
        img_dataset = dataset_module(path='data', resize=args.resize, val_ratio=args.val_ratio)
        train_set, val_set = img_dataset.split_dataset()
    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              num_workers=4,
                              pin_memory=use_cuda,
                              drop_last=True,
                              shuffle=True)
    val_loader = DataLoader(val_set,
                            batch_size=args.batch_size,
                            num_workers=4,
                            pin_memory=use_cuda,
                            drop_last=True,
                            shuffle=True)

    # -- model
    if args.dataset == 'MNISTDataset':
        model = ResNet50(num_classes, channels=1).to('cuda')
    else:
        model = ResNet50(num_classes, channels=3).to('cuda')

    # -- config
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=2)

    # -- logging
    log_dir = os.path.join('log/ResNet', args.dataset)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = SummaryWriter(log_dir=log_dir)
    with open(os.path.join(log_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    pth_dir = os.path.join('resnet_pytorch', args.dataset)
    if not os.path.exists(pth_dir):
        os.makedirs(pth_dir)

    best_val_loss = np.inf

    # -- train
    for epoch in range(args.epochs):
        print(f'EPOCH {epoch+1} training...')
        model.train()
        train_losses = []
        running_loss = 0
        for i, train_batch in tqdm(enumerate(train_loader)):
            train_start = time.time()
            inputs, labels = train_batch
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            train_losses.append(loss.item())

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 0 and i > 0:
                print(f' >>> minibatch {i} -- Training Loss: {running_loss/100:4.5} || Time(s): {time.time()-train_start:4.5}')
                running_loss = 0.0
                avg_loss = sum(train_losses)/len(train_losses)
                logger.add_scalar("Train/loss", avg_loss, epoch * len(train_loader) + i)
        scheduler.step(avg_loss)

        # -- validate

        with torch.no_grad():
            model.eval()
            val_acc_history = []
            val_loss_history = []
            for val_batch in tqdm(val_loader):
                val_start = time.time()
                inputs, labels = val_batch
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                predicted = torch.argmax(outputs, dim=-1)
                loss = criterion(outputs, labels).item()
                acc = (labels == predicted).sum().item()

                val_acc_history.append(acc)
                val_loss_history.append(loss)
            val_acc = np.sum(val_acc_history) / len(val_set)
            val_loss = np.sum(val_loss_history) / len(val_loader)
            print(f'EPOCH {epoch+1} / minibatch {i} -- Validation Accuracy: {100*val_acc:4.8}%, Validation Loss: {val_loss:4.5} || Time(s): {time.time()-val_start:4.5}')
            logger.add_scalar("Val/accuracy", val_acc, epoch)
            logger.add_scalar("Val/loss", val_loss, epoch)
            if val_loss < best_val_loss:
                print(f'Saving model with better loss with {val_loss:4.5}...')
                torch.save(model.state_dict(), f'{pth_dir}/epoch_{epoch+1}_loss_{val_loss}.pth')
                best_val_loss = val_loss

            print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=123, help='random seed (default: 123)')
    parser.add_argument('--dataset', required=True, type=str, help='choose dataset for training (MNIST, CIFAR, ImageNet)')
    parser.add_argument('--resize', required=True, type=int, help='resize image shape for each model')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (difault: 10)')
    parser.add_argument('--batch_size', type=int, default=32, help='choose batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
    args = parser.parse_args()

    train(args)
