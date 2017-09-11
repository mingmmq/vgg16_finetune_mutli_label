import os
from bs4 import BeautifulSoup
import math

rejects2007 = [u'002001', u'007976', u'007953', u'009800', u'001732', u'002465', u'002738', u'004371', u'005181', u'007365', u'008999', u'000222', u'001393', u'001563', u'002931', u'004508', u'006483', u'006939', u'007572', u'007631', u'008042', u'008841', u'008989', u'000091', u'000541', u'000700', u'001112', u'002544', u'003214', u'004748', u'005071', u'005169', u'006766', u'007305', u'007736', u'007932', u'007963', u'008197', u'008315', u'008336', u'008449', u'008784', u'009515', u'003184', u'001683', u'002989', u'005129', u'005191', u'005591', u'005710', u'005923', u'006030', u'007327', u'007959', u'008462', u'008936', u'009828', u'002868', u'003887', u'004686', u'008706', u'008108', u'009596', u'000066', u'000222', u'000288', u'000411', u'000477', u'000688', u'000793', u'000906', u'000967', u'001279', u'001333', u'001434', u'001766', u'001902', u'001950', u'001989', u'002311', u'002362', u'002572', u'002595', u'002625', u'002759', u'003261', u'003484', u'003700', u'003945', u'003974', u'004170', u'004359', u'004387', u'004723', u'005134', u'005173', u'005374', u'005509', u'005542', u'005600', u'005961', u'006229', u'006320', u'006369', u'006874', u'006896', u'007325', u'007497', u'007699', u'007872', u'007911', u'007932', u'007953', u'008449', u'008482', u'008676', u'008771', u'008794', u'009073', u'009214', u'009288', u'009336', u'009469', u'009638', u'009671', u'009762', u'009949', u'003211', u'003367', u'003555', u'004037', u'005605', u'007396', u'008216', u'008806', u'008933', u'001733', u'002845', u'006251', u'009945', u'008665', u'008462']
rejects2007val = [u'000150', u'000251', u'000251', u'000269', u'000374', u'000500', u'000523', u'000591', u'000613', u'000684', u'000702', u'000755', u'000855', u'000935', u'000971', u'001018', u'001142', u'001241', u'001352', u'001444', u'001445', u'001464', u'001472', u'001472', u'001598', u'001691', u'001860', u'001899', u'001964', u'002226', u'002244', u'002290', u'002387', u'002504', u'002504', u'002513', u'002605', u'002613', u'002658', u'002741', u'002859', u'002913', u'002965', u'002965', u'003056', u'003129', u'003170', u'003218', u'003228', u'003311', u'003344', u'003407', u'003419', u'003470', u'003567', u'003567', u'003589', u'003618', u'003636', u'003669', u'003703', u'003711', u'003760', u'003783', u'003924', u'003992', u'004122', u'004203', u'004258', u'004349', u'004380', u'004437', u'004437', u'004459', u'004527', u'004588', u'004655', u'004699', u'004812', u'004953', u'005081', u'005230', u'005326', u'005326', u'005475', u'005485', u'005507', u'005522', u'005554', u'005652', u'005662', u'005768', u'005794', u'005919', u'005940', u'006018', u'006133', u'006235', u'006396', u'006465', u'006497', u'006542', u'006572', u'006588', u'006661', u'006673', u'006803', u'006821', u'006841', u'006908', u'006965', u'007021', u'007059', u'007068', u'007117', u'007461', u'007527', u'007709', u'007856', u'007897', u'008105', u'008105', u'008209', u'008241', u'008444', u'008503', u'008526', u'008557', u'008562', u'008604', u'008717', u'008718', u'008775', u'008836', u'008943', u'008997', u'009113', u'009177', u'009368', u'009456', u'009532', u'009558', u'009617', u'009617', u'009647', u'009758', u'009816', u'009822', u'009822', u'009822', u'009902']

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
                # if count == 64:
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
    from PIL import ImageFont, Image, ImageDraw
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

    font = ImageFont.truetype("simsun.ttf", 15)
    for i in range(12):
        for j in range(12):
            draw.text((i*h_gap + 2,j * v_gap +2), str(j*12 + i), font=font)


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

    # im.show()
    return im


def getImageAndAnnotations(path, last_name):

    main_path = "/".join([path, "ImageSets/Main"])
    image_list = get_image_list(main_path, last_name)
    cat_dict = getCategoryDict(main_path)

    file_obj_pos = {}
    count = 0
    for image in image_list:
        if image in rejects2007 or image in rejects2007val:
            continue

        anno_filepath = "/".join([path, "Annotations", image+".xml"])
        anno = load_annotation(anno_filepath)

        #do detect if the
        obj_pos = get_objects(anno)
        file_obj_pos[image] = obj_pos
        # im = draw_image(path, image)
        # im.save("/".join(["images",image+" " + str(obj_pos)+".jpg"]))

        # this part is for showing only the 10 items
        if count == 64:
            break
        count += 1


    return file_obj_pos


if __name__ == "__main__":
    print(os.curdir)
    # print(os.listdir("VOC2012/ImageSets/Main/"))

