from PIL import Image, ImageEnhance, ImageFile
import random
from torchvision.transforms import ToTensor as torchtotensor
import cv2
import numpy as np
import torch
import torchvision.transforms.functional

MAX_VALUES_BY_DTYPE = {
    np.dtype("uint8"): 255,
    np.dtype("uint16"): 65535,
    np.dtype("uint32"): 4294967295,
    np.dtype("float32"): 1.0,
}


################################iamge################################
class Compose_imglabel(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, label, border=None):
        for t in self.transforms:
            if border is None:
                img, label = t(img, label)
            else:
                img, label, border = t(img, label, border=border)
        if border is None:
            return img, label
        return img, label, border


class Random_horizontal_flip(object):
    def _horizontal_flip(self, img, label, border=None):
        # dsa
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        if border is None:
            return img, label
        border = border.transpose(Image.FLIP_LEFT_RIGHT)
        return img, label, border

    def __init__(self, prob):
        '''
        :param prob: should be (0,1)
        '''
        assert prob >= 0 and prob <= 1, "prob should be [0,1]"
        self.prob = prob

    def __call__(self, img, label, border=None):
        '''
        flip img and label simultaneously
        :param img:should be PIL image
        :param label:should be PIL image
        :return:
        '''
        assert isinstance(img, Image.Image), "should be PIL image"
        assert isinstance(label, Image.Image), "should be PIL image"
        if random.random() < self.prob:
            return self._horizontal_flip(img, label, border)
        else:
            if border is None:
                return img, label
            return img, label, border


class Random_vertical_flip(object):
    def _vertical_flip(self, img, label, border=None):
        # dsaFLIP_TOP_BOTTOM
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        label = label.transpose(Image.FLIP_TOP_BOTTOM)
        if border is None:
            return img, label
        border = border.transpose(Image.FLIP_TOP_BOTTOM)
        return img, label, border

    def __init__(self, prob):
        assert prob >= 0 and prob <= 1, "prob should be [0,1]"
        self.prob = prob

    def __call__(self, img, label, border=None):
        assert isinstance(img, Image.Image), "should be PIL image"
        assert isinstance(label, Image.Image), "should be PIL image"
        if random.random() < self.prob:
            return self._vertical_flip(img, label, border)
        else:
            if border is None:
                return img, label
            return img, label, border


class Random_crop_Resize(object):
    def _randomCrop(self, img, label):
        width, height = img.size
        x, y = random.randint(0, self.crop_size), random.randint(0, self.crop_size)
        region = [x, y, width - x, height - y]
        img, label = img.crop(region), label.crop(region)
        img = img.resize((width, height), Image.BILINEAR)
        label = label.resize((width, height), Image.BILINEAR)
        return img, label

    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, img, label):
        assert img.size == label.size, "img should have the same shape as label"
        return self._randomCrop(img, label)


class Resize1(object):
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, img, label, border=None):
        img = img.resize((self.width, self.height), Image.BILINEAR)
        label = label.resize((self.width, self.height), Image.BILINEAR)
        if border is None:
            return img, label
        border = border.resize((self.width, self.height), Image.BILINEAR)
        return img, label, border


class Normalize11(object):
    def __init__(self, mean, std):
        self.mean, self.std = mean, std

    def __call__(self, img, label, border=None):
        for i in range(3):
            img[:, :, i] -= float(self.mean[i])
        for i in range(3):
            img[:, :, i] /= float(self.std[i])
        if border is None:
            return img, label
        return img, label, border


class random_rotate(object):
    def __init__(self, range=[0, 360], interval=1, p=0.5):
        self.range = range
        self.interval = interval
        self.p = p

    def __call__(self, img, label, border=None):
        rot = (random.randint(*self.range) // self.interval) * self.interval
        rot = rot + 360 if rot < 0 else rot

        if random.random() < self.p:
            base_size = label.size
            img = img.rotate(rot, expand=True)
            label = label.rotate(rot, expand=True)

            img = img.crop(((img.size[0] - base_size[0]) // 2,
                            (img.size[1] - base_size[1]) // 2,
                            (img.size[0] + base_size[0]) // 2,
                            (img.size[1] + base_size[1]) // 2))

            label = label.crop(((label.size[0] - base_size[0]) // 2,
                                (label.size[1] - base_size[1]) // 2,
                                (label.size[0] + base_size[0]) // 2,
                                (label.size[1] + base_size[1]) // 2))
            if border is not None:
                border = border.rotate(rot, expand=True)
                border = border.crop(((border.size[0] - base_size[0]) // 2,
                                      (border.size[1] - base_size[1]) // 2,
                                      (border.size[0] + base_size[0]) // 2,
                                      (border.size[1] + base_size[1]) // 2))
        if border is None:
            return img, label
        return img, label, border


class random_rotate90(object):
    def __init__(self, interval=1, p=0.5):
        self.interval = interval
        self.p = p

    def __call__(self, img, label, border=None):
        rot = random.choice([1, 2, 3]) * 90
        # rot = rot + 360 if rot < 0 else rot

        if random.random() < self.p:
            base_size = label.size
            img = img.rotate(rot, expand=True)
            label = label.rotate(rot, expand=True)

            if border is not None:
                border = border.rotate(rot, expand=True)
        if border is None:
            return img, label
        return img, label, border


class random_scale_crop(object):
    def __init__(self, range=[0.75, 1.25], p=0.5):
        self.range = range
        self.p = p

    def __call__(self, img, label, border=None):
        scale = random.random() * (self.range[1] - self.range[0]) + self.range[0]
        if random.random() < self.p:
            base_size = label.size
            scale_size = tuple((np.array(base_size) * scale).round().astype(int))
            img = img.resize(scale_size)
            label = label.resize(scale_size)

            # 裁剪图像，可以为负数
            img = img.crop(((img.size[0] - base_size[0]) // 2,
                            (img.size[1] - base_size[1]) // 2,
                            (img.size[0] + base_size[0]) // 2,
                            (img.size[1] + base_size[1]) // 2))

            label = label.crop(((label.size[0] - base_size[0]) // 2,
                                (label.size[1] - base_size[1]) // 2,
                                (label.size[0] + base_size[0]) // 2,
                                (label.size[1] + base_size[1]) // 2))
            if border is not None:
                border = border.resize(scale_size)
                border = border.crop(((border.size[0] - base_size[0]) // 2,
                                      (border.size[1] - base_size[1]) // 2,
                                      (border.size[0] + base_size[0]) // 2,
                                      (border.size[1] + base_size[1]) // 2))
        if border is None:
            return img, label
        return img, label, border


class random_translate(object):
    def __init__(self, translate=[0.3, 0.3], p=0.5):
        '''
        For example translate=(a, b),
        then horizontal shift is randomly sampled in the range
        -img_width * a < dx < img_width * a
        and vertical shift is randomly sampled in the range
        -img_height * b < dy < img_height * b.
        '''
        self.translate = translate
        self.p = p

    def __call__(self, img, label, border=None):
        if random.random() < self.p:
            base_size = label.size

            max_dx = round(self.translate[0] * base_size[0])
            max_dy = round(self.translate[1] * base_size[1])
            tx = random.randint(-max_dx, max_dx)
            ty = random.randint(-max_dy, max_dy)
            translations = (tx, ty)

            # img = img.crop((-20, -20, 200, 200))
            h1 = translations[0]
            w1 = translations[1]
            h2 = h1 + base_size[0]
            w2 = w1 + base_size[1]

            img = img.crop((h1, w1, h2, w2))
            label = label.crop((h1, w1, h2, w2))
            if border is None:
                return img, label
            border = border.crop((h1, w1, h2, w2))
            return img, label, border

        if border is None:
            return img, label
        return img, label, border


class random_enhance(object):
    def __init__(self, p=0.5, methods=['contrast', 'brightness', 'sharpness']):
        self.p = p
        self.enhance_method = []
        if 'contrast' in methods:
            self.enhance_method.append(ImageEnhance.Contrast)
        if 'brightness' in methods:
            self.enhance_method.append(ImageEnhance.Brightness)
        if 'sharpness' in methods:
            self.enhance_method.append(ImageEnhance.Sharpness)

    def __call__(self, img, label, border=None):
        np.random.shuffle(self.enhance_method)
        for method in self.enhance_method:
            if np.random.random() < self.p:
                enhancer = method(img)
                factor = float(1 + np.random.random() / 10)
                img = enhancer.enhance(factor)

        if border is None:
            return img, label
        return img, label, border


class random_dilation_erosion:
    def __init__(self, p=0.5, kernel_range=[2, 5]):
        self.p = p,
        self.kernel_range = kernel_range

    def __call__(self, img, label):
        label = np.array(label)
        key = random.random()
        # kernel = np.ones(tuple([np.random.randint(*self.kernel_range)]) * 2, dtype=np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (random.randint(*self.kernel_range),) * 2)
        if key < 1 / 3:
            label = cv2.dilate(label, kernel)
        elif 1 / 3 <= key < 2 / 3:
            label = cv2.erode(label, kernel)

        label = Image.fromarray(label)

        return img, label


class toTensor(object):
    def __init__(self):
        self.totensor = torchtotensor()

    def __call__(self, img, label, border=None):
        img, label = self.totensor(img), self.totensor(label).long()
        if border is None:
            return img, label
        border = self.totensor(border).long()
        return img, label, border


class normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, label, border=None):
        img = np.array(img, dtype=np.float32)
        label = np.array(label, dtype=np.float32)
        img /= 255
        img -= self.mean
        img /= self.std

        label /= 255
        if border is not None:
            border = np.array(border, dtype=np.float32)
            border /= 255
            return img, label, border
        return img, label


class totensor(object):
    def __init__(self):
        pass

    def __call__(self, img, label, border=None):
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img).float()

        label = torch.from_numpy(label)
        label = label.unsqueeze(dim=0)

        if border is not None:
            border = torch.from_numpy(border)
            border = border.unsqueeze(dim=0)
            return img, label, border

        return img, label


################################video################################
class Random_crop_Resize_Video(object):
    def _randomCrop(self, img, label, x, y):
        width, height = img.size
        region = [x, y, width - x, height - y]
        img, label = img.crop(region), label.crop(region)
        img = img.resize((width, height), Image.BILINEAR)
        label = label.resize((width, height), Image.BILINEAR)
        return img, label

    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, imgs, labels):
        res_img = []
        res_label = []
        x, y = random.randint(0, self.crop_size), random.randint(0, self.crop_size)
        for img, label in zip(imgs, labels):
            img, label = self._randomCrop(img, label, x, y)
            res_img.append(img)
            res_label.append(label)
        return res_img, res_label


class Random_horizontal_flip_video(object):

    def __init__(self, prob):
        '''
        :param prob: should be (0,1)
        '''
        assert prob >= 0 and prob <= 1, "prob should be [0,1]"
        self.prob = prob

    def __call__(self, imgs, labels, border=None):
        '''
        flip img and label simultaneously
        :param img:should be PIL image
        :param label:should be PIL image
        :return:
        '''
        if random.random() < self.prob:
            res_img = []
            res_label = []
            res_border = []
            for i, (img, label) in enumerate(zip(imgs, labels)):
                img, label = img.transpose(Image.FLIP_LEFT_RIGHT), label.transpose(Image.FLIP_LEFT_RIGHT)
                res_img.append(img)
                res_label.append(label)
                if border is not None:
                    border1 = border[i]
                    border1 = border1.transpose(Image.FLIP_LEFT_RIGHT)
                    res_border.append(border1)

            if border is None:
                return res_img, res_label
            return res_img, res_label, res_border
        else:
            if border is None:
                return imgs, labels
            return imgs, labels, border


class Random_vertical_flip_video(object):

    def __init__(self, prob):
        '''
        :param prob: should be (0,1)
        '''
        assert prob >= 0 and prob <= 1, "prob should be [0,1]"
        self.prob = prob

    def __call__(self, imgs, labels, border=None):
        '''
        flip img and label simultaneously
        :param img:should be PIL image
        :param label:should be PIL image
        :return:
        '''
        if random.random() < self.prob:
            res_img = []
            res_label = []
            res_border = []
            for i, (img, label) in enumerate(zip(imgs, labels)):
                img, label = img.transpose(Image.FLIP_TOP_BOTTOM), label.transpose(Image.FLIP_TOP_BOTTOM)
                res_img.append(img)
                res_label.append(label)
                if border is not None:
                    border1 = border[i]
                    border1 = border1.transpose(Image.FLIP_TOP_BOTTOM)
                    res_border.append(border1)

            if border is None:
                return res_img, res_label
            return res_img, res_label, res_border
        else:
            if border is None:
                return imgs, labels
            return imgs, labels, border



class Resize_video(object):
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, imgs, labels, border=None):
        res_img = []
        res_label = []
        res_border = []
        for i, (img, label) in enumerate(zip(imgs, labels)):
            res_img.append(img.resize((self.width, self.height), Image.BILINEAR))
            res_label.append(label.resize((self.width, self.height), Image.BILINEAR))
            if border is not None:
                border1 = border[i]
                res_border.append(border1.resize((self.width, self.height), Image.BILINEAR))
        if border is None:
            return res_img, res_label
        return res_img, res_label, res_border



class random_rotate_video(object):
    def __init__(self, range=[0, 360], interval=1, p=0.5):
        self.range = range
        self.interval = interval
        self.p = p

    def __call__(self, imgs, labels, border=None):
        rot = (random.randint(*self.range) // self.interval) * self.interval
        rot = rot + 360 if rot < 0 else rot

        if random.random() < self.p:
            res_img, res_label = [], []
            res_border = []
            for i, (img, label) in enumerate(zip(imgs, labels)):
                base_size = label.size
                img = img.rotate(rot, expand=True)
                label = label.rotate(rot, expand=True)

                img = img.crop(((img.size[0] - base_size[0]) // 2,
                                (img.size[1] - base_size[1]) // 2,
                                (img.size[0] + base_size[0]) // 2,
                                (img.size[1] + base_size[1]) // 2))

                label = label.crop(((label.size[0] - base_size[0]) // 2,
                                    (label.size[1] - base_size[1]) // 2,
                                    (label.size[0] + base_size[0]) // 2,
                                    (label.size[1] + base_size[1]) // 2))
                res_img.append(img)
                res_label.append(label)

                if border is not None:
                    border1 = border[i]
                    border1 = border1.rotate(rot, expand=True)
                    border1 = border1.crop(((border1.size[0] - base_size[0]) // 2,
                                            (border1.size[1] - base_size[1]) // 2,
                                            (border1.size[0] + base_size[0]) // 2,
                                            (border1.size[1] + base_size[1]) // 2))
                    res_border.append(border1)

            if border is None:
                return res_img, res_label
            return res_img, res_label, res_border
        else:
            if border is None:
                return imgs, labels
            return imgs, labels, border


class random_rotate90_video(object):
    def __init__(self, interval=1, p=0.5):
        self.interval = interval
        self.p = p

    def __call__(self, imgs, labels, border=None):
        rot = random.choice([1, 2, 3]) * 90

        if random.random() < self.p:
            res_img, res_label = [], []
            res_border = []
            for i, (img, label) in enumerate(zip(imgs, labels)):
                base_size = label.size
                img = img.rotate(rot, expand=True)
                label = label.rotate(rot, expand=True)

                res_img.append(img)
                res_label.append(label)

                # print(len(border), len(imgs), len(labels))
                if border is not None:
                    border1 = border[i]
                    border1 = border1.rotate(rot, expand=True)
                    res_border.append(border1)

            if border is None:
                return res_img, res_label
            return res_img, res_label, res_border
        else:
            if border is None:
                return imgs, labels
            return imgs, labels, border


class random_translate_video(object):
    def __init__(self, translate=[0.3, 0.3], p=0.5):
        '''
        For example translate=(a, b),
        then horizontal shift is randomly sampled in the range
        -img_width * a < dx < img_width * a
        and vertical shift is randomly sampled in the range
        -img_height * b < dy < img_height * b.
        '''
        self.translate = translate
        self.p = p

    def __call__(self, imgs, labels, border=None):
        if random.random() < self.p:

            base_size = labels[0].size
            max_dx = round(self.translate[0] * base_size[0])
            max_dy = round(self.translate[1] * base_size[1])
            tx = random.randint(-max_dx, max_dx)
            ty = random.randint(-max_dy, max_dy)
            translations = (tx, ty)

            h1 = translations[0]
            w1 = translations[1]
            h2 = h1 + base_size[0]
            w2 = w1 + base_size[1]

            imgs_res, labels_res = [], []
            border_res = []
            for i, (img, label) in enumerate(zip(imgs, labels)):
                img = img.crop((h1, w1, h2, w2))
                label = label.crop((h1, w1, h2, w2))
                imgs_res.append(img)
                labels_res.append(label)
                if border is not None:
                    border1 = border[i]
                    border1 = border1.crop((h1, w1, h2, w2))
                    border_res.append(border1)
            if border is None:
                return imgs_res, labels_res
            return imgs_res, labels_res, border_res
        if border is None:
            return imgs, labels
        return imgs, labels, border


class random_enhance_video(object):
    def __init__(self, p=0.5, methods=['contrast', 'brightness', 'sharpness']):
        self.p = p
        self.enhance_method = []
        if 'contrast' in methods:
            self.enhance_method.append(ImageEnhance.Contrast)
        if 'brightness' in methods:
            self.enhance_method.append(ImageEnhance.Brightness)
        if 'sharpness' in methods:
            self.enhance_method.append(ImageEnhance.Sharpness)

    def __call__(self, imgs, labels, border=None):
        np.random.shuffle(self.enhance_method)
        imgs_res = []
        for img in imgs:
            for method in self.enhance_method:
                if np.random.random() < self.p:
                    enhancer = method(img)
                    factor = float(1 + np.random.random() / 10)
                    img = enhancer.enhance(factor)
            imgs_res.append(img)

        if border is None:
            return imgs_res, labels
        return imgs_res, labels, border


class random_scale_crop_video(object):
    def __init__(self, range=[0.75, 1.25], p=0.5):
        self.range = range
        self.p = p

    def __call__(self, imgs, labels, border=None):
        scale = random.random() * (self.range[1] - self.range[0]) + self.range[0]
        if random.random() < self.p:
            base_size = labels[0].size
            scale_size = tuple((np.array(base_size) * scale).round().astype(int))

            imgs_res, labels_res = [], []
            border_res = []
            for i, (img, label) in enumerate(zip(imgs, labels)):
                img = img.resize(scale_size)
                label = label.resize(scale_size)

                img = img.crop(((img.size[0] - base_size[0]) // 2,
                                (img.size[1] - base_size[1]) // 2,
                                (img.size[0] + base_size[0]) // 2,
                                (img.size[1] + base_size[1]) // 2))

                label = label.crop(((label.size[0] - base_size[0]) // 2,
                                    (label.size[1] - base_size[1]) // 2,
                                    (label.size[0] + base_size[0]) // 2,
                                    (label.size[1] + base_size[1]) // 2))
                imgs_res.append(img)
                labels_res.append(label)
                if border is not None:
                    border1 = border[i]
                    border1 = border1.resize(scale_size)
                    border1 = border1.crop(((border1.size[0] - base_size[0]) // 2,
                                            (border1.size[1] - base_size[1]) // 2,
                                            (border1.size[0] + base_size[0]) // 2,
                                            (border1.size[1] + base_size[1]) // 2))
                    border_res.append(border1)
            if border is None:
                return imgs_res, labels_res
            return imgs_res, labels_res, border_res

        if border is None:
            return imgs, labels
        return imgs, labels, border


class normalize_video(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, imgs, labels, border=None):
        # image, gt = sample['image'], sample['gt']
        imgs_res, labels_res = [], []
        border_res = []
        for i, (img, label) in enumerate(zip(imgs, labels)):
            img, label = np.array(img, dtype=np.float32), np.array(label, dtype=np.float32)
            img /= 255
            img -= self.mean
            img /= self.std

            label /= 255
            imgs_res.append(img)
            labels_res.append(label)
            if border is not None:
                border1 = border[i]
                border1 = np.array(border1, dtype=np.float32)
                border1 /= 255
                border_res.append(border1)
        if border is not None:
            return imgs_res, labels_res, border_res

        return imgs_res, labels_res


class totensor_video(object):
    def __init__(self):
        pass

    def __call__(self, imgs, labels, border=None):

        imgs_res, labels_res = [], []
        border_res = []
        for i, (img, label) in enumerate(zip(imgs, labels)):
            img = img.transpose((2, 0, 1))
            img = torch.from_numpy(img).float()
            label = torch.from_numpy(label)
            label = label.unsqueeze(dim=0)
            imgs_res.append(img)
            labels_res.append(label)

            if border is not None:
                border1 = border[i]
                border1 = torch.from_numpy(border1)
                border1 = border1.unsqueeze(dim=0)
                border_res.append(border1)

        if border is not None:
            return imgs_res, labels_res, border_res

        return imgs_res, labels_res


class Normalize_video(object):
    def __init__(self, mean, std):
        self.mean, self.std = mean, std

    def __call__(self, imgs, labels, border=None):
        res_img = []
        for img in imgs:
            for i in range(3):
                img[:, :, i] -= float(self.mean[i])
            for i in range(3):
                img[:, :, i] /= float(self.std[i])
            res_img.append(img)
        if border is None:
            return res_img, labels
        return res_img, labels, border


class toTensor_video(object):
    def __init__(self):
        self.totensor = torchtotensor()

    def __call__(self, imgs, labels, border=None):
        res_img = []
        res_label = []
        res_border = []
        for i, (img, label) in enumerate(zip(imgs, labels)):
            img, label = self.totensor(img), self.totensor(label).long()
            res_img.append(img)
            res_label.append(label)
            if border is not None:
                border1 = border[i]
                border1 = self.totensor(border1).long()
                res_border.append(border1)
        if border is None:
            return res_img, res_label
        return res_img, res_label, res_border
