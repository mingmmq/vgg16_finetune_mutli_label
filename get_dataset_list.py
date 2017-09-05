import os
from bs4 import BeautifulSoup
import math

rejects2007 = [u'002001', u'007976', u'007953', u'009800', u'001732', u'002465', u'002738', u'004371', u'005181', u'007365', u'008999', u'000222', u'001393', u'001563', u'002931', u'004508', u'006483', u'006939', u'007572', u'007631', u'008042', u'008841', u'008989', u'000091', u'000541', u'000700', u'001112', u'002544', u'003214', u'004748', u'005071', u'005169', u'006766', u'007305', u'007736', u'007932', u'007963', u'008197', u'008315', u'008336', u'008449', u'008784', u'009515', u'003184', u'001683', u'002989', u'005129', u'005191', u'005591', u'005710', u'005923', u'006030', u'007327', u'007959', u'008462', u'008936', u'009828', u'002868', u'003887', u'004686', u'008706', u'008108', u'009596', u'000066', u'000222', u'000288', u'000411', u'000477', u'000688', u'000793', u'000906', u'000967', u'001279', u'001333', u'001434', u'001766', u'001902', u'001950', u'001989', u'002311', u'002362', u'002572', u'002595', u'002625', u'002759', u'003261', u'003484', u'003700', u'003945', u'003974', u'004170', u'004359', u'004387', u'004723', u'005134', u'005173', u'005374', u'005509', u'005542', u'005600', u'005961', u'006229', u'006320', u'006369', u'006874', u'006896', u'007325', u'007497', u'007699', u'007872', u'007911', u'007932', u'007953', u'008449', u'008482', u'008676', u'008771', u'008794', u'009073', u'009214', u'009288', u'009336', u'009469', u'009638', u'009671', u'009762', u'009949', u'003211', u'003367', u'003555', u'004037', u'005605', u'007396', u'008216', u'008806', u'008933', u'001733', u'002845', u'006251', u'009945', u'008665', u'008462']

def getImageAndLabels(path, last_name):
    label_path = "/".join([path, "ImageSets/Main/"])
    files = [file for file in os.listdir(label_path) if file.__contains__(last_name)]
    files = sorted(files)
    print(files)

    i = 0
    dictionary = {}
    for file in files:
        print(str(i) +": " + file + "\n")

        with open(os.path.join(label_path,file)) as f:
            # lines = f.read()
            # lines = [os.path.join('VOC2012', 'JPEGImages', line.split()[0] + ".jpg")  for line in f.read().splitlines() if  line.__contains__(" 1")]
            lines = f.read().splitlines()
            count = 0
            for img_path in lines:

                # if count == 128:
                #     break
                # count += 1

                if img_path.__contains__(" -1"):
                    continue

                img_path = os.path.join(path, 'JPEGImages', img_path.split()[0] + ".jpg")

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
    print(image_sets)
    d = {}
    i = 0
    for cate in image_sets:
        d[cate] = i
        i += 1

    return d



# annotation operations
def load_annotation(anno_filename):
    xml = ""
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

def get_objects(anno):
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

        y = math.floor(ycen * 12)
        x = math.floor(xcen * 12)
        position = y*12 + x

        obj_name = obj.findChildren('name')[0].contents[0]

        if obj_pos_dict.__contains__(obj_name):
            obj_pos_dict[obj_name].append(position)
        else:
            obj_pos_dict[obj_name] = [position]


    return obj_pos_dict

def draw_image(path, image_name):
    anno_path =  "/".join([path, "Annotations", image_name+".xml"])
    anno = load_annotation(anno_path)
    size = anno.find('size')
    width = float(size.findChild('width').contents[0])
    height = float(size.findChild('height').contents[0])

    image_path = "/".join([path, "JPEGImages", image_name+".jpg"])
    from PIL import Image, ImageDraw
    im = Image.open(image_path)
    draw = ImageDraw.Draw(im)
    h_gap = width/12
    v_gap = height/12
    for i in range(1,12):
        draw.line((i*h_gap,0, i*h_gap,height), fill=128)
        draw.line((0,i*v_gap, width, i*v_gap), fill=128)

    objs = anno.findAll('object')
    for obj in objs:

        bbox = obj.findChildren('bndbox')[0]
        xmin = int(bbox.findChildren('xmin')[0].contents[0])
        ymin = int(bbox.findChildren('ymin')[0].contents[0])
        xmax = int(bbox.findChildren('xmax')[0].contents[0])
        ymax = int(bbox.findChildren('ymax')[0].contents[0])
        xcen = (xmin + xmax) / 2.0
        ycen = (ymin + ymax) / 2.0
        draw.line((xcen,0, xcen,height), fill=255)
        draw.line((0,ycen, width,ycen), fill=255)

    im.show()


def getImageAndAnnotations(path, last_name):

    main_path = "/".join([path, "ImageSets/Main"])
    image_list = get_image_list(main_path, last_name)
    cat_dict = getCategoryDict(main_path)

    file_obj_pos = {}
    for image in image_list:
        if image in rejects2007:
            continue

        anno_filepath = "/".join([path, "Annotations", image+".xml"])
        anno = load_annotation(anno_filepath)

        #do detect if the
        # draw_image(path, image)
        obj_pos = get_objects(anno)
        file_obj_pos[image] = obj_pos

    return file_obj_pos


if __name__ == "__main__":
    print(os.curdir)
    # print(os.listdir("VOC2012/ImageSets/Main/"))

