import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models


class FinetuneVgg16Model(nn.Module):
    def __init__(self, original_model, num_classes):
        super(FinetuneVgg16Model, self).__init__()

        self.features = original_model.features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        self.modelName = 'vgg16'

        # Freeze those weights
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x):
        f = self.features(x)
        f = f.view(f.size(0), -1)
        y = self.classifier(f)
        return y

def main():
    num_classes = 10
    print("num_classess = '{}'".format(num_classes))


    #__dict__ a dictionary or other mapping object used to store an object's
    #writable attributes
    # here I don't have to use, but it can be used to avoid using the if statements
    original_model = models.__dict__["vgg16"](pretrained=True)
    model = FinetuneVgg16Model(original_model, num_classes)

    cudnn.benchmark = True

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    cifar = datasets.CIFAR10("../../data/cifar",
                             transforms.Compose([
                                 transforms.RandomSizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 normalize,
                             ]),
                             download=True
                             )

    train_loader = torch.utils.data.DataLoader(
        cifar,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    for i, (image, target) in enumerate(train_loader):
        print(i, image, target)


    pass

if __name__ == "__main__":
    main()
