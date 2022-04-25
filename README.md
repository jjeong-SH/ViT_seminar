# ViT_seminar
training 3 different models from scratch - for image processing trasformers seminar


## directory tree
```
├── data
│   ├── CIFAR
│   ├── CLS-LOC
│   └── MNIST
├── resnet_pytorch
│   └── resnet.py
├── efficientnet_pytorch
│   ├── model.py
│   └── utils.py
├── vit_pytorch
│   └── vit.py
├── log
│   ├── EfficientNet
│   │   ├── CIFARDataset
│   │   ├── ImageNetDataset
│   │   └── MNISTDataset
│   ├── ResNet
│   │   ├── CIFARDataset
│   │   ├── ImageNetDataset
│   │   └── MNISTDataset
│   └── ViT
│       ├── CIFARDataset
│       ├── ImageNetDataset
│       └── MNISTDataset
├── download.py
├── test.py
├── dataset.py
├── train_resnet.py
├── train_efcntnet.py
└── train_vit.py
```

## data
* MNIST (10 classes)
* CIFAR10 (10 classes)
* ImageNet ILSVRC2012 (20 classes - randomly selected distinctive classes)

*not included in git repo due to size limits*

![seminar_dataset](https://user-images.githubusercontent.com/80621384/165035796-dff9d647-0237-454e-aa33-a9d7d6f5c253.png)


## models
* ResNet50
* EfficientNetB3
* ViT

![seminar_models](https://user-images.githubusercontent.com/80621384/165036241-33d25e38-eee4-4c2c-8223-b644bbf89aab.png)


## train
```
python {train_resnet.py/train_efcntnet.py/train_vit.py} --seed 123 --dataset {MNISTDataset/CIFARDataset/ImageNetDataset} --resize 224 --val_ratio 0.2 --epochs 20 --batch_size 16
```

## log
```
tensorboard --logdir=./log/{modelname}/{datasetname}
```
