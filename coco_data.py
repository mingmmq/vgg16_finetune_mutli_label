from __future__ import absolute_import
from keras.preprocessing import image
import numpy as np
from keras.utils import np_utils
import cv2

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from pycocotools.coco import COCO
import math



def show_image(image_data, lables=""):
    arr = np.ascontiguousarray(image_data.transpose(1, 2, 0))
    img = Image.fromarray(arr, 'RGB')
    font = ImageFont.truetype("/Library/Fonts/Arial.ttf", 10)
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), lables, (255, 255, 0), font=font)
    img.show()

def load_data(image_path=""):
    if image_path == "":
        print("data_path required: coco image path required")
        exit()
    (x_train, y_train) = load_data_by_type(image_path, "train")
    (x_test, y_test)  = load_data_by_type(image_path, "val")

    return (x_train, y_train), (x_test, y_test)

def check_in_the_same_grid(points, dim):
    the_dict = {}
    for point in points:
        x = int(math.floor(point[0]  * dim))
        y = int(math.floor(point[1] * dim))
        key = x + y*dim
        if key not in the_dict:
            the_dict[key] = 0;
        the_dict[key] += 1

    for value in the_dict.values():
        if value > 1:
            return True

    return False

def check_grid(coco, img, dim):
    annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
    anns = coco.loadAnns(annIds)

    width = img['width']
    height = img['height']

    cat_points = {}
    points = []
    fname =img['file_name']
    for anno in anns:
        bbox = anno['bbox']
        xmin = bbox[0]
        ymin = bbox[1]
        iw = bbox[2]
        ih = bbox[3]
        xcen = (xmin + iw / 2.0) / width
        ycen = (ymin + ih / 2.0) / height
        cat_id = anno['category_id']

        if cat_id not in cat_points:
            cat_points[cat_id] = []
        cat_points[cat_id].append([xcen, ycen])


    if cat_points.keys().__len__() is 0:
        return False

    # print anno.findChild('filename').contents[0]
    for key, value in cat_points.items():
        if check_in_the_same_grid(value, dim) is True:
            # print "have objects in the same grid"
            return False

    # print "no objects in the same grid"
    return True

def load_data_by_type(path, type):
    #this is used the set the grid line numbers, and the rejected images are listed by another program
    grid_rows = 4
    if type == "train":
        type_path = "train2014"
    else:
        type_path = "val2014"
    annFile = '%s/annotations/instances_%s.json' % (path, type_path)
    coco = COCO(annFile)
    data = coco.imgs

    num_samples = 0
    for image_id, img_info in data.items():
        # #next two lines are used for filter out those only with one label
        # if len(data[key]) > 1:
        #     continue

        # no need to check the grid in the detection task
        # if check_grid(coco, img_info, grid_rows):
        num_samples += 1

    # num_train_samples = len(data.keys())
    # num_train_samples = 32

    print(num_samples)

    x_train = np.zeros((num_samples, 3, 224, 224), dtype='uint8')
    # y_train = np.zeros((num_train_samples, 20), dtype='uint8')
    i = 0
    labels = []
    image_names = []

    # In what order will the key be iterated? the order in linux is different from in macos
    files = []
    for image_id, img_info in data.items():

        image_path = "/".join([path, "images",type_path, img_info["file_name"]])

        image_names.append(img_info["file_name"])
        img = image.load_img(image_path)
        d = image.img_to_array(img, data_format="channels_first").astype(dtype="uint8")
        dr = cv2.resize(d.transpose(1, 2, 0), (224, 224)).transpose(2, 0, 1)
        x_train[i,:,:,:] = dr

        labels.append(object_positions(coco, img_info, grid_rows))
        files.append(img_info["file_name"])

        if i + 1 == num_samples:
            break
        i += 1

    y_train = to_categoricals_v1(path, labels, 90)

    print("samples counts:", num_samples)
    return (x_train, y_train)


def object_positions(coco, img, grid_rows):
    annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
    anns = coco.loadAnns(annIds)

    width = img['width']
    height = img['height']

    objects_pos = {}
    for anno in anns:
        bbox = anno['bbox']
        xmin = bbox[0]
        ymin = bbox[1]
        iw = bbox[2]
        ih = bbox[3]
        xcen = (xmin + iw / 2.0) / width
        ycen = (ymin + ih / 2.0) / height
        cat_id = anno['category_id']

        y = math.floor(ycen * grid_rows)
        x = math.floor(xcen * grid_rows)
        position = y*grid_rows + x

        if cat_id not in objects_pos:
            objects_pos[cat_id] = [position]
        else:
            objects_pos[cat_id].append(position)

    return objects_pos

def to_categoricals_v1(coco, y, num_classes):

    n = len(y)

    categorical = np.zeros((n, num_classes)).astype('float64')
    for i in range(0, n):
        lists = [x - 1 for x in list(y[i].copy().keys())]
        categorical[i,  lists] = 1

    return categorical


def to_categoricals(coco, y, num_classes, grid_rows):
    # y = np.array(y, dtype='int')
    # if not num_classes:
    #     num_classes = np.max(y) + 1

    n = len(y)
    categorical = np.zeros((n, num_classes * grid_rows * grid_rows)).astype('float64')

    for i in range(0, n):
        # print("\n", files[i], y[i])
        for key in y[i]:
            lists = [int(j + grid_rows * grid_rows * (key - 1)) for j in y[i][key]]
            # print(key, lists)
            categorical[i,  lists] = 1

    return categorical


if __name__ == "__main__":
    load_data()

