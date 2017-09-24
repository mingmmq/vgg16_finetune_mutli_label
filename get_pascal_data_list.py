import os
from bs4 import BeautifulSoup
import math

rejects2007 = [u'002001', u'007976', u'007953', u'009800', u'001732', u'002465', u'002738', u'004371', u'005181', u'007365', u'008999', u'000222', u'001393', u'001563', u'002931', u'004508', u'006483', u'006939', u'007572', u'007631', u'008042', u'008841', u'008989', u'000091', u'000541', u'000700', u'001112', u'002544', u'003214', u'004748', u'005071', u'005169', u'006766', u'007305', u'007736', u'007932', u'007963', u'008197', u'008315', u'008336', u'008449', u'008784', u'009515', u'003184', u'001683', u'002989', u'005129', u'005191', u'005591', u'005710', u'005923', u'006030', u'007327', u'007959', u'008462', u'008936', u'009828', u'002868', u'003887', u'004686', u'008706', u'008108', u'009596', u'000066', u'000222', u'000288', u'000411', u'000477', u'000688', u'000793', u'000906', u'000967', u'001279', u'001333', u'001434', u'001766', u'001902', u'001950', u'001989', u'002311', u'002362', u'002572', u'002595', u'002625', u'002759', u'003261', u'003484', u'003700', u'003945', u'003974', u'004170', u'004359', u'004387', u'004723', u'005134', u'005173', u'005374', u'005509', u'005542', u'005600', u'005961', u'006229', u'006320', u'006369', u'006874', u'006896', u'007325', u'007497', u'007699', u'007872', u'007911', u'007932', u'007953', u'008449', u'008482', u'008676', u'008771', u'008794', u'009073', u'009214', u'009288', u'009336', u'009469', u'009638', u'009671', u'009762', u'009949', u'003211', u'003367', u'003555', u'004037', u'005605', u'007396', u'008216', u'008806', u'008933', u'001733', u'002845', u'006251', u'009945', u'008665', u'008462']
rejects2007val = [u'000150', u'000251', u'000251', u'000269', u'000374', u'000500', u'000523', u'000591', u'000613', u'000684', u'000702', u'000755', u'000855', u'000935', u'000971', u'001018', u'001142', u'001241', u'001352', u'001444', u'001445', u'001464', u'001472', u'001472', u'001598', u'001691', u'001860', u'001899', u'001964', u'002226', u'002244', u'002290', u'002387', u'002504', u'002504', u'002513', u'002605', u'002613', u'002658', u'002741', u'002859', u'002913', u'002965', u'002965', u'003056', u'003129', u'003170', u'003218', u'003228', u'003311', u'003344', u'003407', u'003419', u'003470', u'003567', u'003567', u'003589', u'003618', u'003636', u'003669', u'003703', u'003711', u'003760', u'003783', u'003924', u'003992', u'004122', u'004203', u'004258', u'004349', u'004380', u'004437', u'004437', u'004459', u'004527', u'004588', u'004655', u'004699', u'004812', u'004953', u'005081', u'005230', u'005326', u'005326', u'005475', u'005485', u'005507', u'005522', u'005554', u'005652', u'005662', u'005768', u'005794', u'005919', u'005940', u'006018', u'006133', u'006235', u'006396', u'006465', u'006497', u'006542', u'006572', u'006588', u'006661', u'006673', u'006803', u'006821', u'006841', u'006908', u'006965', u'007021', u'007059', u'007068', u'007117', u'007461', u'007527', u'007709', u'007856', u'007897', u'008105', u'008105', u'008209', u'008241', u'008444', u'008503', u'008526', u'008557', u'008562', u'008604', u'008717', u'008718', u'008775', u'008836', u'008943', u'008997', u'009113', u'009177', u'009368', u'009456', u'009532', u'009558', u'009617', u'009617', u'009647', u'009758', u'009816', u'009822', u'009822', u'009822', u'009902']


rejects_7 = [u'000657', u'002001', u'007976', u'002513', u'005175', u'008101', u'008236', u'009085', u'009461', u'001654', u'002069', u'004359', u'007108', u'007953', u'008478', u'000755', u'006206', u'006661', u'002664', u'003416', u'006848', u'007194', u'007431', u'008588', u'009800', u'001413', u'001860', u'001964', u'002762', u'002984', u'004349', u'005485', u'006012', u'008943', u'001732', u'002465', u'002738', u'003337', u'004371', u'005181', u'007365', u'007521', u'008999', u'006046', u'006357', u'006673', u'000222', u'000367', u'000753', u'001393', u'001563', u'001580', u'002253', u'002518', u'002931', u'003577', u'004742', u'006483', u'006503', u'006939', u'007410', u'007572', u'007740', u'007751', u'008042', u'008841', u'008967', u'008989', u'009497', u'009585', u'000269', u'000564', u'001490', u'002030', u'002226', u'002613', u'002941', u'002965', u'003056', u'003207', u'003783', u'005662', u'005919', u'006398', u'007021', u'007141', u'007461', u'008319', u'009647', u'009902', u'000477', u'003407', u'000091', u'000320', u'000541', u'000620', u'000700', u'000977', u'001112', u'001492', u'002180', u'002490', u'002544', u'003214', u'003420', u'003713', u'004019', u'004136', u'004303', u'004691', u'004828', u'005071', u'005918', u'006369', u'006766', u'006931', u'007205', u'007305', u'007736', u'007883', u'007932', u'007963', u'008197', u'008296', u'008315', u'008336', u'008449', u'008706', u'008784', u'009318', u'009515', u'009845', u'009879', u'000303', u'000329', u'000653', u'000663', u'000855', u'000935', u'001093', u'001371', u'001386', u'001445', u'001472', u'001561', u'001801', u'001899', u'002135', u'002244', u'002504', u'002606', u'003344', u'003636', u'004073', u'004186', u'004203', u'004488', u'004494', u'004539', u'004727', u'005110', u'006018', u'006330', u'006396', u'006588', u'006783', u'006821', u'007004', u'007068', u'007383', u'007427', u'007527', u'007650', u'007950', u'008105', u'008359', u'008503', u'008562', u'008739', u'008848', u'008892', u'008911', u'009006', u'009015', u'009254', u'000592', u'000597', u'000865', u'001345', u'001683', u'002156', u'002645', u'002778', u'005129', u'005153', u'005191', u'005591', u'005730', u'005920', u'005923', u'006605', u'006658', u'007006', u'007213', u'007327', u'007959', u'008216', u'008462', u'008936', u'009526', u'009638', u'009828', u'000249', u'000492', u'000591', u'001018', u'001027', u'001887', u'001901', u'002456', u'002636', u'003078', u'003335', u'004105', u'004655', u'004785', u'005701', u'005723', u'005794', u'006220', u'006286', u'006530', u'006628', u'007799', u'007826', u'007865', u'008717', u'008997', u'009051', u'009550', u'009617', u'009758', u'000827', u'002868', u'003887', u'004537', u'004686', u'006170', u'008202', u'000464', u'000834', u'002940', u'004585', u'005326', u'006841', u'007528', u'008345', u'007166', u'001042', u'008115', u'005351', u'008040', u'000667', u'000760', u'006862', u'001532', u'009596', u'003567', u'006216', u'009034', u'009822', u'000066', u'000083', u'000288', u'000394', u'000411', u'000654', u'000688', u'000709', u'000739', u'000793', u'000828', u'000829', u'000906', u'000943', u'000967', u'001248', u'001279', u'001333', u'001414', u'001434', u'001455', u'001499', u'001711', u'001766', u'001828', u'001864', u'001902', u'001904', u'001950', u'001989', u'002193', u'002194', u'002241', u'002311', u'002362', u'002472', u'002572', u'002595', u'002625', u'002870', u'003003', u'003261', u'003369', u'003484', u'003496', u'003576', u'003628', u'003797', u'003856', u'003879', u'003945', u'003974', u'004058', u'004091', u'004170', u'004244', u'004387', u'004392', u'004563', u'004571', u'004648', u'004715', u'004718', u'004857', u'004902', u'005134', u'005173', u'005219', u'005273', u'005311', u'005374', u'005451', u'005509', u'005542', u'005600', u'005786', u'005961', u'006038', u'006103', u'006225', u'006243', u'006320', u'006381', u'006459', u'006486', u'006551', u'006784', u'006874', u'006896', u'006919', u'007325', u'007466', u'007497', u'007578', u'007631', u'007699', u'007814', u'007836', u'007872', u'007911', u'008079', u'008108', u'008232', u'008263', u'008397', u'008398', u'008482', u'008529', u'008663', u'008676', u'008748', u'008771', u'008794', u'008879', u'008960', u'009045', u'009073', u'009123', u'009214', u'009283', u'009469', u'009671', u'009703', u'009762', u'009792', u'009949', u'000110', u'000125', u'000150', u'000169', u'000177', u'000251', u'000374', u'000500', u'000523', u'000545', u'000613', u'000684', u'000702', u'000854', u'000926', u'000971', u'001028', u'001241', u'001259', u'001311', u'001352', u'001691', u'001849', u'001958', u'002174', u'002290', u'002385', u'002387', u'002425', u'002563', u'002693', u'002741', u'002859', u'002886', u'002913', u'003044', u'003065', u'003129', u'003170', u'003218', u'003228', u'003311', u'003313', u'003470', u'003589', u'003638', u'003648', u'003654', u'003669', u'003703', u'003711', u'003760', u'003992', u'004122', u'004258', u'004437', u'004459', u'004588', u'004630', u'004699', u'004812', u'004859', u'004953', u'005014', u'005036', u'005081', u'005116', u'005161', u'005230', u'005278', u'005343', u'005398', u'005475', u'005507', u'005522', u'005554', u'005652', u'005747', u'005799', u'005863', u'006029', u'006071', u'006133', u'006150', u'006161', u'006198', u'006235', u'006346', u'006465', u'006497', u'006542', u'006584', u'006625', u'006647', u'006666', u'006670', u'006803', u'006908', u'006965', u'007052', u'007059', u'007117', u'007296', u'007314', u'007346', u'007709', u'007856', u'007897', u'007928', u'007984', u'008069', u'008224', u'008241', u'008444', u'008514', u'008526', u'008592', u'008604', u'008692', u'008752', u'008775', u'008836', u'009064', u'009141', u'009224', u'009303', u'009349', u'009368', u'009433', u'009448', u'009456', u'009532', u'009558', u'009649', u'009712', u'009726', u'009946', u'003367', u'003555', u'003699', u'003834', u'004037', u'005605', u'007396', u'008199', u'008806', u'008933', u'009887', u'001142', u'001464', u'002658', u'003093', u'003419', u'003924', u'004275', u'005894', u'008557', u'009686', u'000416', u'001594', u'001733', u'002845', u'003874', u'006251', u'008150', u'009945', u'000588', u'001598', u'002267', u'002986', u'004527', u'005395', u'007217', u'008292', u'008942', u'009816', u'004386', u'000675', u'007915', u'009388', u'000626', u'002767', u'006323']

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
            # lines = [os.path.join('VOC2012', 'JPEGImages', line.split()[0] + ".jpg")  for line in f.read().splitlines() if  line.__contains__(" 1")]
            lines = f.read().splitlines()
            count = 0
            for img_path in lines:
                # if count == 64:
                #     break
                # count += 1

                if " -1" in img_path:
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

    image_path = "/".join([path, "JPEGImages", image_name+".jpg"])
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

def getImageAndAnnotations(path, last_name, grid_rows):

    main_path = "/".join([path, "ImageSets/Main"])
    image_list = get_image_list(main_path, last_name)
    cat_dict = getCategoryDict(main_path)

    objects_count = 0
    file_obj_pos = {}
    count = 0
    for image in image_list:

        if not pass_grid_check(path, image, grid_rows):
            print(image)
            # draw_image(path, image, grid_rows)
            continue

        anno_filepath = "/".join([path, "Annotations", image+".xml"])
        anno = load_annotation(anno_filepath)

        #do detect if the
        obj_pos = get_objects(anno, grid_rows=grid_rows)
        file_obj_pos[image] = obj_pos
        objects_count += obj_pos.__len__()
        # im = draw_image(path, image)
        # im.save("/".join(["images",image+" " + str(obj_pos)+".jpg"]))

        # this part is for showing only the 10 items
        # if count == 1023:
        #     break
        # count += 1


    print("total objects: ", objects_count)
    return file_obj_pos


if __name__ == "__main__":
    print(os.curdir)
    # print(os.listdir("VOC2012/ImageSets/Main/"))
