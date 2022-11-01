import os
import numpy as np
# from utils.preprocess import *
from all_config.config_pretraining import config


def save_sqc():
    
    video_dataset_root = config.video_dataset_root
    video_dataset_list = ["CVC-ClinicDB-612", "CVC-ColonDB-300"]
    video_filelist = {}
    for video_name in video_dataset_list:
        video_root = os.path.join(video_dataset_root, video_name)
        cls_list = os.listdir(video_root)
        for cls in cls_list:
            video_filelist[video_name + '-' + cls] = {
                'img': [],
                'gt': [],
                'border': [],
            }
            cls_path = os.path.join(video_root, cls)
            cls_img_path = os.path.join(cls_path, "Frame")
            cls_gt_path = os.path.join(cls_path, "GT")
            cls_border_path = os.path.join(cls_path, "border")

            img_list = os.listdir(cls_img_path)
            img_list.sort()
            for filename in img_list:
                video_filelist[video_name + '-' + cls]['img'].append(os.path.join(cls_img_path, filename))

            gt_list = os.listdir(cls_gt_path)
            gt_list.sort()
            for filename in gt_list:
                video_filelist[video_name + '-' + cls]['gt'].append(os.path.join(cls_gt_path, filename))

            border_list = os.listdir(cls_border_path)
            border_list.sort()
            for filename in border_list:
                video_filelist[video_name + '-' + cls]['border'].append(os.path.join(cls_border_path, filename))

    np.save('./sqc_pathlist.npy', video_filelist)
    print('finish!')

save_sqc()
