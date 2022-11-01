import numpy as np
import cv2
import os
from PIL import Image
from all_config.config_pretraining import config

def test():
    img_li = []
    img_path1 = config.video_dataset_root + "/CVC-ColonDB-300/Train/2/Frame/68.jpg"
    img_li.append(config.video_dataset_root + "/CVC-ColonDB-300/Train/11/Frame/300.jpg")
    img_li.append(config.video_dataset_root + "/CVC-ColonDB-300/Train/11/Frame/299.jpg")
    img_li.append(config.video_dataset_root + "/CVC-ColonDB-300/Train/11/Frame/298.jpg")
    img_li.append(config.video_dataset_root + "/CVC-ColonDB-300/Train/11/Frame/297.jpg")
    img_li.append(config.video_dataset_root + "/CVC-ColonDB-300/Train/11/Frame/296.jpg")

    # name2  = self.color1[idx%len(self.color1)] if np.random.rand()<0.7 else self.color2[idx%len(self.color2)]
    image1 = cv2.imread(img_path1)
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2LAB)

    img = []
    for path in img_li:
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        img.append(image)

    mean, std = image1.mean(axis=(0, 1), keepdims=True), image1.std(axis=(0, 1), keepdims=True)

    mean2 = np.zeros((len(img), *mean.shape))
    std2 = np.zeros((len(img), *mean.shape))
    for i in range(len(img)):
        mean2[i, ::], std2[i, ::] = img[i].mean(axis=(0, 1), keepdims=True), img[i].std(axis=(0, 1), keepdims=True)


    mean2 = np.mean(mean2, axis=0)
    std2 = np.mean(std2, axis=0)

    image = ((image1 - mean) / std * std2 + mean2)

    image[image > 255] = 255
    image[image < 0] = 0
    # cv2.normalize(image, image, 0, 255, cv2.NORM_MINMAX)
    # print(image.min(), image.max())
    image = np.uint8(image)
    # print(image.min(), image.max())

    image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("p2.jpg", image)
    print('finish!')

def testPIL():
    img_li = []
    img_path1 = config.video_dataset_root + "/CVC-ColonDB-300/Train/2/Frame/68.jpg"
    img_li.append(config.video_dataset_root + "/CVC-ColonDB-300/Train/11/Frame/300.jpg")
    img_li.append(config.video_dataset_root + "/CVC-ColonDB-300/Train/11/Frame/299.jpg")
    img_li.append(config.video_dataset_root + "/CVC-ColonDB-300/Train/11/Frame/298.jpg")
    img_li.append(config.video_dataset_root + "/CVC-ColonDB-300/Train/11/Frame/297.jpg")
    img_li.append(config.video_dataset_root + "/CVC-ColonDB-300/Train/11/Frame/296.jpg")

    # name2  = self.color1[idx%len(self.color1)] if np.random.rand()<0.7 else self.color2[idx%len(self.color2)]
    image1 = Image.open(img_path1)
    image1 = np.array(image1)

    # image1 = cv2.imread(img_path1)
    image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2LAB)

    img = []
    for path in img_li:
        image = Image.open(path)
        image = np.array(image)
        # image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        img.append(image)

    mean, std = image1.mean(axis=(0, 1), keepdims=True), image1.std(axis=(0, 1), keepdims=True)

    mean2 = np.zeros((len(img), *mean.shape))
    std2 = np.zeros((len(img), *mean.shape))
    for i in range(len(img)):
        mean2[i, ::], std2[i, ::] = img[i].mean(axis=(0, 1), keepdims=True), img[i].std(axis=(0, 1), keepdims=True)


    mean2 = np.mean(mean2, axis=0)
    std2 = np.mean(std2, axis=0)

    image = ((image1 - mean) / std * std2 + mean2)

    image[image > 255] = 255
    image[image < 0] = 0
    # cv2.normalize(image, image, 0, 255, cv2.NORM_MINMAX)
    # print(image.min(), image.max())
    image = np.uint8(image)

    image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
    image = Image.fromarray(image)

    image.save('p2.jpg')
    # cv2.imwrite("p2.jpg", image)
    print('finish!')

def save():
    # CVC - ClinicDB - 612\Train\1\Frame
    video_dataset_root = config.video_dataset_root
    video_dataset_list = ["CVC-ClinicDB-612", "CVC-ColonDB-300"]
    video_filelist = {}
    for video_name in video_dataset_list:
        video_root = os.path.join(video_dataset_root, video_name, 'Train')
        cls_list = os.listdir(video_root)
        for cls in cls_list:
            video_filelist[video_name + '-' + cls] = []
            cls_path = os.path.join(video_root, cls)
            cls_img_path = os.path.join(cls_path, "Frame")
            tmp_list = os.listdir(cls_img_path)
            tmp_list.sort()
            for filename in tmp_list:
                video_filelist[video_name + '-' + cls].append(os.path.join(cls_img_path, filename))

    mean_std = {}
    for k, v in video_filelist.items():
        img_list = v
        img = []
        mean_std[k] = []
        for path in img_list:
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            img.append(image)

        mean2 = np.zeros((len(img), 1, 1, 3))
        std2 = np.zeros((len(img), 1, 1, 3))
        for i in range(len(img)):
            mean2[i, ::] = img[i].mean(axis=(0, 1), keepdims=True)
            std2[i, ::] = img[i].std(axis=(0, 1), keepdims=True)

        mean2 = np.mean(mean2, axis=0)
        std2 = np.mean(std2, axis=0)

        mean_std[k].append(mean2)
        mean_std[k].append(std2)

    np.save('./mean.npy', mean_std)


def savePIL():
    # CVC - ClinicDB - 612\Train\1\Frame
    video_dataset_root = config.video_dataset_root
    video_dataset_list = ["CVC-ClinicDB-612", "CVC-ColonDB-300"]
    video_filelist = {}
    for video_name in video_dataset_list:
        video_root = os.path.join(video_dataset_root, video_name)
        cls_list = os.listdir(video_root)
        for cls in cls_list:
            video_filelist[video_name + '-' + cls] = []
            cls_path = os.path.join(video_root, cls)
            cls_img_path = os.path.join(cls_path, "Frame")
            tmp_list = os.listdir(cls_img_path)
            tmp_list.sort()
            for filename in tmp_list:
                video_filelist[video_name + '-' + cls].append(os.path.join(cls_img_path, filename))

    mean_std = {}
    for k, v in video_filelist.items():
        img_list = v
        img = []
        mean_std[k] = []
        for path in img_list:
            image = Image.open(path)
            image = np.array(image)
            # image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            img.append(image)

        mean2 = np.zeros((len(img), 1, 1, 3))
        std2 = np.zeros((len(img), 1, 1, 3))
        for i in range(len(img)):
            mean2[i, ::] = img[i].mean(axis=(0, 1), keepdims=True)
            std2[i, ::] = img[i].std(axis=(0, 1), keepdims=True)

        mean2 = np.mean(mean2, axis=0)
        std2 = np.mean(std2, axis=0)

        mean_std[k].append(mean2)
        mean_std[k].append(std2)

    np.save('./mean.npy', mean_std)
    print('finish!')

savePIL()

