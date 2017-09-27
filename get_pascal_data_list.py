import os
from bs4 import BeautifulSoup
import math

img_in_one_cell = ['2008_000151',  '2008_000255',  '2008_000915', '2008_001161']
big = ['2008_001852', '2008_001881','2008_001882','2008_001894']
cell_centre =  ['2008_001783', '2008_001921', '2008_001926', '2008_002056']
cell_edge = ['2008_001787','2008_001789','2008_001852','2008_001888','2008_001903']

def getImageAndLabels(path, last_name):
    label_path = "/".join([path, "ImageSets/Main/"])
    files = [file for file in os.listdir(label_path) if file.__contains__(last_name)]
    files = sorted(files)
    print(files)

    i = 0
    dictionary = {}
    for file in files:
        # print(str(i) +": " + file + "\n")

        with open(os.path.join(label_path,file)) as f:
            # lines = f.read()
            # lines = [os.path.join('VOC2012', 'JPEGImages', line.split()[0] + ".train")  for line in f.read().splitlines() if  line.__contains__(" 1")]
            lines = f.read().splitlines()
            count = 0
            for img_path in lines:
                # if count == 64:
                #     break
                # count += 1

                if " -1" in img_path:
                    continue

                img_path = os.path.join(path, 'JPEGImages', img_path.split()[0] + ".train")

                if dictionary.__contains__(img_path):
                    dictionary[img_path].append(i)
                else:
                    dictionary[img_path] = [i]
        i += 1

    # print(dictionary)
    return dictionary, files

def getCategoryDict(main_path):
    all_files = os.listdir(main_path)
    image_sets = sorted(list(set([filename.replace('.txt', '').strip().split('_')[0] for filename in all_files if
                                  filename.__contains__("_train.txt")])))
    # print(image_sets)
    d = {}
    i = 0
    for cate in image_sets:
        d[cate] = i
        i += 1

    return d


# annotation operations
def load_annotation(anno_filename):
    with open(anno_filename) as f:
        xml = f.readlines()
    xml = ''.join([line.strip('\t') for line in xml])
    return BeautifulSoup(xml, "html5lib")


def get_image_list(main_path, last_name):

    image_files_name_path = "/".join([main_path, last_name[1:]])
    print(image_files_name_path)
    with open(image_files_name_path) as f:
        image_names = f.read().splitlines()
    return image_names

def get_objects(anno, grid_rows):
    size = anno.find('size')
    width = float(size.findChild('width').contents[0])
    height = float(size.findChild('height').contents[0])
    # print width, height

    points = []
    objs = anno.findAll('object')
    obj_pos_dict = {}
    for obj in objs:

        bbox = obj.findChildren('bndbox')[0]
        xmin = int(bbox.findChildren('xmin')[0].contents[0])
        ymin = int(bbox.findChildren('ymin')[0].contents[0])
        xmax = int(bbox.findChildren('xmax')[0].contents[0])
        ymax = int(bbox.findChildren('ymax')[0].contents[0])
        xcen = (xmin + xmax) / 2.0 / width
        ycen = (ymin + ymax) / 2.0 / height
        points.append([xcen, ycen])

        y = math.floor(ycen * grid_rows)
        x = math.floor(xcen * grid_rows)
        position = y*grid_rows + x

        obj_name = obj.findChildren('name')[0].contents[0]

        if obj_pos_dict.__contains__(obj_name):
            obj_pos_dict[obj_name].append(position)
        else:
            obj_pos_dict[obj_name] = [position]


    return obj_pos_dict

def draw_image(path, image_name, grid_rows):
    from PIL import ImageFont, Image, ImageDraw
    anno_path =  "/".join([path, "Annotations", image_name+".xml"])
    anno = load_annotation(anno_path)
    size = anno.find('size')
    width = float(size.findChild('width').contents[0])
    height = float(size.findChild('height').contents[0])

    image_path = "/".join([path, "JPEGImages", image_name+".train"])
    from PIL import Image, ImageDraw
    im = Image.open(image_path)
    draw = ImageDraw.Draw(im)
    h_gap = width/grid_rows
    v_gap = height/grid_rows
    for i in range(1,grid_rows):
        draw.line((i*h_gap,0, i*h_gap,height), fill=128)
        draw.line((0,i*v_gap, width, i*v_gap), fill=128)

    font = ImageFont.truetype("simsun.ttf", 15)
    for i in range(grid_rows):
        for j in range(grid_rows):
            draw.text((i*h_gap + 2,j * v_gap +2), str(j*grid_rows + i), font=font)


    objs = anno.findAll('object')
    for obj in objs:

        bbox = obj.findChildren('bndbox')[0]
        xmin = int(bbox.findChildren('xmin')[0].contents[0])
        ymin = int(bbox.findChildren('ymin')[0].contents[0])
        xmax = int(bbox.findChildren('xmax')[0].contents[0])
        ymax = int(bbox.findChildren('ymax')[0].contents[0])
        xcen = (xmin + xmax) / 2.0
        ycen = (ymin + ymax) / 2.0
        draw.ellipse((xcen-4, ycen-4, xcen +4,ycen +4), fill = 'blue', outline ='blue')

    im.show()
    return im

def not_in_the_same_grid(points, dim):
    the_dict = {}
    for point in points:
        x = int(math.floor(point[0]  * dim))
        y = int(math.floor(point[1] * dim))
        key = str(x) + ',' + str(y)
        if key not in the_dict:
            the_dict[key] = 1
        else:
            the_dict[key] += 1

    for value in the_dict.values():
        if value > 1:
            return True

    return False



def pass_grid_check(path, img, dim):
    anno_path =  "/".join([path, "Annotations", img+".xml"])
    anno = load_annotation(anno_path)
    size = anno.find('size')
    width = float(size.findChild('width').contents[0])
    height = float(size.findChild('height').contents[0])

    points_in_cat = {}
    objs = anno.findAll('object')
    for obj in objs:
        name_tag = obj.findChild('name')
        fname = anno.findChild('filename').contents[0]

        bbox = obj.findChildren('bndbox')[0]
        xmin = int(bbox.findChildren('xmin')[0].contents[0])
        ymin = int(bbox.findChildren('ymin')[0].contents[0])
        xmax = int(bbox.findChildren('xmax')[0].contents[0])
        ymax = int(bbox.findChildren('ymax')[0].contents[0])
        xcen = (xmin + xmax) / 2.0 / width
        ycen = (ymin + ymax) / 2.0 / height

        if name_tag not in points_in_cat:
            points_in_cat[name_tag] = []
        points_in_cat[name_tag].append([xcen, ycen])
            # print xcen, ycen

    # print anno.findChild('filename').contents[0]
    for cat, centers in points_in_cat.items():
        if not_in_the_same_grid(centers, dim):
            # print "have objects in the same grid"
            return False

    # print "no objects in the same grid"
    return True

def getImageAndAnnotations(path, last_name, grid_rows, set_type="all"):

    main_path = "/".join([path, "ImageSets/Main"])
    image_list = get_image_list(main_path, last_name)
    cat_dict = getCategoryDict(main_path)

    objects_count = 0
    file_obj_pos = {}
    count = 0
    for image in image_list:


        if not pass_grid_check(path, image, grid_rows):
            continue

        # draw_image(path, image, grid_rows)

        if set_type == "in_one_cell":
            if file_obj_pos.__len__() == img_in_one_cell.__len__():
                break;
            if image not in img_in_one_cell:
                continue
        elif set_type == "big_than_cell":
            if image not in big:
                continue
        elif set_type == "cell_centre":
            if image not in cell_centre:
                continue
        elif set_type == "cell_edge":
            if image not in cell_edge:
                continue
        else:
            if count == 16:
                break
            count += 1


        anno_filepath = "/".join([path, "Annotations", image+".xml"])
        anno = load_annotation(anno_filepath)

        obj_pos = get_objects(anno, grid_rows=grid_rows)
        file_obj_pos[image] = obj_pos
        objects_count += obj_pos.__len__()

        # this part is for showing only the 10 items
        # if count == 16:
        #     break
        # count += 1


    print("total objects: ", objects_count)
    return file_obj_pos


if __name__ == "__main__":
    print(os.curdir)
    # print(os.listdir("VOC2012/ImageSets/Main/"))

