from bs4 import BeautifulSoup
import pandas as pd
import numpy as np


class Pascal_Multi_Label:
    pascal_class_dic = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4,
                        'bus': 5, 'car': 6, 'cat': 7, 'chair': 8, 'cow': 9,
                        'diningtable': 10, 'dog': 11, 'horse': 12, 'motorbike': 13, 'person': 14,
                        'pottedplant': 15, 'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19}

    def __init__(self, image_name_file, anno_path):
        self.image_name_file = image_name_file
        self.anno_path = anno_path

    # annotation operations
    def load_annotation(self, anno_filename):
        with open(anno_filename) as f:
            xml = f.readlines()
        xml = ''.join([line.strip('\t') for line in xml])
        return BeautifulSoup(xml, "lxml")

    def getLabel(self, anno_file):
        anno = self.load_annotation(anno_file)
        objs = anno.findAll('object')
        label = []
        for obj in objs:
            label_item = self.pascal_class_dic[obj.findChildren('name')[0].contents[0]]
            if not label.__contains__(label_item):
                label.append(label_item)

        return label

    def getImgAndLabels(self):
        # iterate through image file
        with open(self.image_name_file) as f:
            imgs = f.read().splitlines()
        self.labels = []
        for img_name in imgs:
            anno_file = "/".join([self.anno_path, img_name + ".xml"])
            self.labels.append(self.getLabel(anno_file))

        self.img_names = [img + ".jpg" for img in imgs]

        return self.img_names, self.labels

    def toCsv(self, csv_file):
        tags_string = []
        for tags in self.labels:
            tag_string = " ".join([str(tag) for tag in tags])
            tags_string.append(tag_string)

        df = pd.DataFrame({'img':self.img_names, 'tags': tags_string})
        df.to_csv(csv_file)

    def fromCsv(self, csv_file):
        df = pd.read_csv(csv_file)
        self.img_names = df['img']
        self.labels = []
        for tag in df['tags'].str.split().tolist():
            self.labels.append(map(int, tag))

        return self.img_names, self.labels


if __name__ == "__main__":
    pascal = Pascal_Multi_Label("/Users/qianminming/Github/data/pascal/VOCdevkit/VOC2007/ImageSets/Main/train.txt",
                      "/Users/qianminming/Github/data/pascal/VOCdevkit/VOC2007/Annotations")
    pascal.getImgAndLabels()
    pascal.toCsv("train_voc2007.csv")

    pascal = Pascal_Multi_Label("/Users/qianminming/Github/data/pascal/VOCdevkit/VOC2007/ImageSets/Main/val.txt",
                                "/Users/qianminming/Github/data/pascal/VOCdevkit/VOC2007/Annotations")
    pascal.getImgAndLabels()
    pascal.toCsv("val_voc2007.csv")


