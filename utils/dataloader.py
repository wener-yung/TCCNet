import os
from torch.utils.data import Dataset
from utils.preprocess import *
from PIL import Image
import torch
import numpy as np

"""
class Pretrain(Dataset)
def get_pretrain_dataset()
class VideoDataset(Dataset)
def get_video_dataset()
"""

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

class Test_Dataset(Dataset):
    def __init__(self, root, testsets, config):
        self.config = config
        self.time_clips = self.config.video_time_clips
        self.memory_size = self.config.memory_size
        self.video_test_list = []
        self.label_test_list = []

        data_dic = test_dic(root, testsets)
        for testset in testsets:
            video_dic = data_dic[testset]

            self.video_filelist = video_dic['Frame']
            self.label_filelist = video_dic['GT']
            cls_list = list(self.video_filelist.keys())

            for cls in cls_list:
                v_li = self.video_filelist[cls]
                l_li = self.label_filelist[cls]

                for begin in range(0, len(v_li), self.memory_size):
                    v_clips = []
                    l_clips = []
                    for t in range(self.memory_size):
                        if begin + t >= len(v_li):
                            v_clips.insert(0, v_li[begin - 1 - (begin + t - len(v_li))])
                            l_clips.insert(0, l_li[begin - 1 - (begin + t - len(l_li))])
                        else:
                            v_clips.append(v_li[begin + t])
                            l_clips.append(l_li[begin + t])
                    self.video_test_list.append(v_clips)
                    self.label_test_list.append(l_clips)

        self.img_label_transform = Compose_imglabel([
            Resize1(self.config.size[0], self.config.size[1]),
            normalize(mean, std),
            totensor(),
        ])

    def __getitem__(self, idx):
        img_path_li = self.video_test_list[idx]
        label_path_li = self.label_test_list[idx]
        img_li = []
        label_li = []
        for idx, (img_path, label_path) in enumerate(zip(img_path_li, label_path_li)):
            img = Image.open(img_path).convert('RGB')
            label = Image.open(label_path).convert('L')
            img, label = self.img_label_transform(img, label)
            img_li.append(img)
            label_li.append(label)
        data = {
            "img": torch.stack(img_li, dim=0),
            "mask": torch.stack(label_li, dim=0),
            "path": img_path_li
        }
        return data

    def __len__(self):
        return len(self.video_test_list)


class PesudoVideo(Dataset):
    """
    Pre-training on pseudo sequences.
    For a training clip, the first frame is sampled from the labeled frames
    and the rest are generated by applying random affine transforms on the
    first frame, such as translation, zooming, cropping, flip and
    rotation.
    """

    def __init__(self, video_dataset_list, config, getall=False):
        super(PesudoVideo, self).__init__()
        self.config = config
        self.time_clips = self.config.video_time_clips
        self.memory_size = self.config.memory_size
        self.video_train_list = []
        self.label_train_list = []
        self.border_train_list = []

        data_dic = np.load(config.sqc_path, allow_pickle=True).item()
        for cls in list(data_dic.keys()):
            video_dic = data_dic[cls]
            v_li = video_dic['img']
            l_li = video_dic['gt']
            b_li = video_dic['border']

            for begin in range(0, len(v_li), self.time_clips):
                self.video_train_list.append(v_li[begin])
                self.label_train_list.append(l_li[begin])
                self.border_train_list.append(b_li[begin])
                if getall:
                    for t in range(begin + 1, min(begin + self.time_clips, len(v_li))):
                        self.video_train_list.append(v_li[t])
                        self.label_train_list.append(l_li[t])
                        self.border_train_list.append(b_li[t])

            if getall:
                for t in range(self.memory_size - 1):
                    self.video_train_list.append(v_li[t])
                    self.label_train_list.append(l_li[t])
                    self.border_train_list.append(b_li[t])

        self.img_label_transform = Compose_imglabel([
            normalize_video(mean, std),
            totensor_video()
        ])
        self.affine_transform = Compose_imglabel([
            # rotation, sheering, zooming, translation, and cropping
            random_translate(translate=[0.2, 0.2], p=0.5),
            random_scale_crop(range=[0.75, 1.25], p=0.5),
            Random_horizontal_flip(prob=0.5),
            Random_vertical_flip(prob=0.5),
            random_rotate(range=[0, 90], p=0.5),
            random_enhance(p=0.5)
        ])
        self.affine_transform2 = Compose_imglabel([
            # rotation, sheering, zooming, translation, and cropping
            random_translate(translate=[0.3, 0.3], p=0.7),
            random_scale_crop(range=[0.75, 1.25], p=0.7),
            Random_horizontal_flip(prob=0.5),
            Random_vertical_flip(prob=0.5),
            random_rotate(range=[0, 359], p=0.7),
            random_enhance(p=0.5)
        ])

    def __getitem__(self, idx):
        img_path = self.video_train_list[idx]
        label_path = self.label_train_list[idx]
        border_path = self.border_train_list[idx]

        img = Image.open(img_path).convert('RGB')
        label = Image.open(label_path).convert('L')
        border = Image.open(border_path).convert('L')
        img = img.resize((self.config.size[0], self.config.size[1]), Image.BILINEAR)
        label = label.resize((self.config.size[0], self.config.size[1]), Image.BILINEAR)
        border = border.resize((self.config.size[0], self.config.size[1]), Image.BILINEAR)

        img_li = [img]
        label_li = [label]
        border_li = [border]

        img11, label11, border11 = self.affine_transform(img, label, border)
        img_li.append(img11)
        label_li.append(label11)
        border_li.append(border11)

        img22, label22, border22 = self.affine_transform2(img, label, border)
        img_li.append(img22)
        label_li.append(label22)
        border_li.append(border22)

        img_li, label_li, border_li = self.img_label_transform(img_li, label_li, border_li)

        data = {"img": torch.stack(img_li, dim=0),
                "mask": torch.stack(label_li, dim=0),
                "border": torch.stack(border_li, dim=0),
                "path": img_path}
        return data

    def __len__(self):
        return len(self.video_train_list)


class RandomVideo(Dataset):
    """
    Main-training on real sequences.
    For a training clip, the first frame is sampled from the labeled frames
    and works as the reference frame. The rest unlabeled frames are
    randomly selected from the same sequence in temporal order.
    """

    def __init__(self, video_dataset_list, config, getall=False, transform=None, time_interval=1):
        super(RandomVideo, self).__init__()
        self.config = config
        self.video_dataset_list = video_dataset_list
        self.time_clips = self.config.video_time_clips
        self.memory_size = self.config.memory_size
        self.mean_dic = np.load(config.mean_dic_path, allow_pickle=True).item()

        self.video_train_list = None
        self.label_train_list = None
        self.border_train_list = None
        self.shufle_data()

        self.img_label_transform = transform

    def shufle_data(self):
        self.video_train_list = []
        self.label_train_list = []
        self.border_train_list = []

        data_dic = np.load(self.config.sqc_path, allow_pickle=True).item()
        for cls in list(data_dic.keys()):
            video_dic = data_dic[cls]
            v_li = video_dic['img']
            l_li = video_dic['gt']
            b_li = video_dic['border']

            for begin in range(0, len(v_li), self.time_clips):
                for i in range((len(v_li) - begin) // 4 + 1):
                    img_li = [v_li[begin]]
                    label_li = [l_li[begin]]
                    border_li = [b_li[begin]]

                    selected_list = range(begin+1, len(v_li))
                    if len(selected_list) >= self.memory_size - 1:
                        select = random.sample(selected_list, self.memory_size - 1)
                    else:
                        select = random.sample(range(0, len(v_li)), self.memory_size - 1)
                    select.sort()
                    for idx in select:
                        img_li.append(v_li[idx])
                        label_li.append(l_li[idx])
                        border_li.append(b_li[idx])

                    self.video_train_list.append(img_li)
                    self.label_train_list.append(label_li)
                    self.border_train_list.append(border_li)

    def __getitem__(self, idx):
        img_label_li = self.video_train_list[idx]
        gt_label_li = self.label_train_list[idx]
        bor_label_li = self.border_train_list[idx]
        img_li = []
        label_li = []
        bor_li = []
        for idx, (img_path, label_path, bor_path) in enumerate(zip(img_label_li, gt_label_li, bor_label_li)):
            img = Image.open(img_path).convert('RGB')
            label = Image.open(label_path).convert('L')
            border = Image.open(bor_path).convert('L')
            img_li.append(img)
            label_li.append(label)
            bor_li.append(border)

        # color_exchange
        if np.random.rand() < 0.5:
            img_cvt = []
            random_color = self.mean_dic[np.random.choice(list(self.mean_dic.keys()))]
            mean2, std2 = random_color[0], random_color[1]
            for img in img_li:
                img = np.array(img)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
                mean1 = img.mean(axis=(0, 1), keepdims=True)
                std1 = img.std(axis=(0, 1), keepdims=True)

                image = (img - mean1) / std1 * std2 + mean2
                image[image > 255] = 255
                image[image < 0] = 0
                image = cv2.cvtColor(np.uint8(image), cv2.COLOR_LAB2RGB)
                img = Image.fromarray(image)
                img_cvt.append(img)
            img_li = img_cvt

        img_trans, label_trans, bor_trans = [], [], []
        for idx, (img, label, bor) in enumerate(zip(img_li, label_li, bor_li)):
            img, label, bor = self.img_label_transform(img, label, bor)
            img_trans.append(img)
            label_trans.append(label)
            bor_trans.append(bor)

        data = {
            "img": torch.stack(img_trans, dim=0),
            "mask": torch.stack(label_trans, dim=0),
            'border': torch.stack(bor_trans, dim=0),
            "path": img_label_li
        }
        return data

    def __len__(self):
        return len(self.video_train_list)


def train_dic(root, video_dataset_list):
    data_dic = {}
    for video_name in video_dataset_list:
        video_root = os.path.join(root, video_name)
        cls_list = os.listdir(video_root)
        cls_list.sort()
        video_filelist = {}
        label_filelist = {}
        border_filelist = {}
        for cls in cls_list:
            video_filelist[cls] = []
            label_filelist[cls] = []
            border_filelist[cls] = []
            cls_path = os.path.join(video_root, cls)
            cls_img_path = os.path.join(cls_path, "Frame")
            cls_label_path = os.path.join(cls_path, "GT")
            cls_border_path = os.path.join(cls_path, "border")
            tmp_list = os.listdir(cls_img_path)
            tmp_list.sort()
            for filename in tmp_list:
                video_filelist[cls].append(
                    os.path.join(cls_img_path, filename)
                )
                border_filelist[cls].append(
                    os.path.join(cls_border_path, filename)
                )
                if '612' in video_name:
                    label_filelist[cls].append(
                        os.path.join(cls_label_path, filename.replace(".bmp", ".tif"))
                    )
                elif '300' in video_name:
                    label_filelist[cls].append(
                        os.path.join(cls_label_path, filename)
                    )
        data_dic[video_name] = {'Frame': video_filelist,
                                'GT': label_filelist,
                                'border': border_filelist
                                }
    return data_dic


def test_dic(root, video_dataset_list):
    data_dic = {}

    for testset in video_dataset_list:
        video_root = os.path.join(root, testset, 'Frame')
        gt_root = os.path.join(root, testset, 'GT')
        cls_list = os.listdir(video_root)
        cls_list.sort()
        video_filelist = {}
        label_filelist = {}
        for cls in cls_list:
            video_filelist[cls] = []
            label_filelist[cls] = []
            frame_cls_path = os.path.join(video_root, cls)
            gt_cls_path = os.path.join(gt_root, cls)
            frame_tmp_list = os.listdir(frame_cls_path)
            frame_tmp_list.sort()
            gt_tmp_list = os.listdir(gt_cls_path)
            gt_tmp_list.sort()

            for filename1 in frame_tmp_list:
                video_filelist[cls].append(
                    os.path.join(frame_cls_path, filename1))
            for filename2 in gt_tmp_list:
                label_filelist[cls].append(
                    os.path.join(gt_cls_path, filename2))

        data_dic[testset] = {'Frame': video_filelist,
                                'GT': label_filelist}
    return data_dic


def get_pesudo_video_dataset(config):
    train_loader = PesudoVideo(config.video_dataset_list, config, config.getall)
    test_loader = Test_Dataset(config.video_testset_root, config.test_dataset_list, config)

    return train_loader, test_loader


def get_random_video_dataset(config):
    trsf_main = Compose_imglabel([
        Resize1(config.size[0], config.size[1]),
        random_translate(translate=[0.2, 0.2], p=0.5),
        random_scale_crop(range=[0.5, 1.25], p=0.5),
        Random_horizontal_flip(0.5),
        Random_vertical_flip(0.5),
        random_rotate(range=[0, 359], p=0.7),
        random_enhance(p=0.5),
        normalize(mean, std),
        totensor()
    ])
    train_loader = RandomVideo(config.video_dataset_list, config, transform=trsf_main)
    test_loader = Test_Dataset(config.video_testset_root, config.test_dataset_list, config)

    return train_loader, test_loader