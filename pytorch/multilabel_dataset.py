from torch.utils.data.dataset import Dataset
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import os
import pandas as pd
import torch
from pytorch.pascal_multi_label import Pascal_Multi_Label

class MultiLabelDataset(Dataset):
    def __init__(self, img_name_file, anno_path, img_path, transform=None, csv_file=None):

        self.mlb = MultiLabelBinarizer()
        self.img_path = img_path
        self.transform = transform

        if csv_file:
            pascal = Pascal_Multi_Label(img_name_file, anno_path)
            self.X_train, tags = pascal.fromCsv(csv_file)
            self.y_train = self.mlb.fit_transform(tags).astype(np.float32)
        else:
            pascal = Pascal_Multi_Label(img_name_file, anno_path)
            self.X_train, tags = pascal.getImgAndLabels()
            self.y_train = self.mlb.fit_transform(tags).astype(np.float32)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path, self.X_train[index]))
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = torch.from_numpy(self.y_train[index])
        return img, label

    def __len__(self):
        return len(self.X_train)

if __name__=="__main__":
    dataset = MultiLabelDataset("/Users/qianminming/Github/data/pascal/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt",
                      "/Users/qianminming/Github/data/pascal/VOCdevkit/VOC2007/Annotations",
                      "/Users/qianminming/Github/data/pascal/VOCdevkit/VOC2007/JPEGImages",
                      "NotNone",
                      "voc2007.csv")


