import os
import cv2
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset

CLASS_PREFIXS = {
    "n01443537" : 0, #goldfish, Carassius auratus
    "n01498041" : 1, #stingray
    "n01582220" : 2, #magpie
    "n01614925" : 3, #bald eagle, American eagle, Haliaeetus leucocephalus
    "n01644373" : 4, #tree frog, tree-frog
    "n01667114" : 5, #mud turtle
    "n01704323" : 6, #triceratops
    "n01728572" : 7, #thunder snake, worm snake, Carphophis amoenus
    "n01770393" : 8, #scorpion
    "n01775062" : 9, #wolf spider, hunting spider
    "n01806143" : 10, #peacock
    "n01910747" : 11, #jellyfish
    "n02007558" : 12, #flamingo
    "n02077923" : 13, #sea lion
    "n02091831" : 14, #Saluki, gazelle hound
    "n02129165" : 15, #lion, king of beasts, Panthera leo
    "n02206856" : 16, #bee
    "n02328150" : 17, #Angora, Angora rabbit
    "n02391049" : 18, #zebra
    "n02480855" : 19, #gorilla, Gorilla gorilla

}


# PATH = './data/CLS-LOC/train'
class ImageNetDataset(Dataset):
    def __init__(self, path, resize, num_classes=10, val_ratio=0.2):
        self.path = path + '/CLS-LOC/train'
        self.img_list = []
        self.labels = []
        self.num_classes = num_classes
        self.val_ratio = val_ratio
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.Resize((resize, resize)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()
        ])
        self.setup_dataset()

    def setup_dataset(self):
        folder_list = os.listdir(self.path)
        for i, folder in enumerate(folder_list):
            #print(folder)
            for files in os.listdir(os.path.join(self.path, folder)):
                _file, ext = os.path.splitext(files)
                filename, _ = _file.split("_")
                file_class = filename.split("/")[-1]

                self.labels.append(CLASS_PREFIXS[file_class])
                self.img_list.append(os.path.join(self.path, folder, files))

    def set_transform(self, transform):
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = self.img_list[idx]
        img = cv2.imread(img, cv2.IMREAD_COLOR)
        #print(img)
        label = self.get_label(idx)
        img = self.transform(img)

        return img, label

    def get_label(self, idx):
        return self.labels[idx]

    def split_dataset(self):
        n_val = int(len(self) * self.val_ratio)
        n_train = len(self) - n_val
        train_set, val_set = torch.utils.data.random_split(self, [n_train, n_val])
        return train_set, val_set


# PATH = './data/MNIST'
def MNISTDataset(path, resize, val_ratio=0.2):
    path = path + '/MNIST'
    transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()
            #transforms.Lambda(lambda x: x.repeat(3, 1, 1))  should match to 3 channel
        ])
    dataset = torchvision.datasets.MNIST(root=path, train=True, download=True, transform=transform)
    n_val = int(len(dataset) * val_ratio)
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])
    return train_set, val_set


# PATH = './data/CIFAR'
def CIFARDataset(path, resize, val_ratio=0.2):
    path = path + '/CIFAR'
    transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()
        ])
    dataset = torchvision.datasets.CIFAR10(root=path, train=True, download=True, transform=transform)
    n_val = int(len(dataset) * val_ratio)
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])
    return train_set, val_set






