import os

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


if __name__ == "__main__":
    print(os.curdir)
    # print(os.listdir("VOC2012/ImageSets/Main/"))

