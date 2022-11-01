from utils.preprocess import *
from PIL import Image
import torch
import numpy as np
import torch.nn.functional as F
import math
from utils.tools import backup, visualize
import numpy as np

stride = 32
crop = 192
base_size = 352
myResize = Resize1(base_size, base_size)
memory_size = 3
statistics = torch.load("/remote-home/share/20-lixiaotong-20210240210/code/PNS-Net-main/utils/statistics.pth")


def getbackground(path, clip=0):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    trans = Compose_imglabel([
        Resize1(352, 352),
        normalize(mean, std),
        totensor(),
    ])

    path_dic = np.load(path, allow_pickle=True).item()
    select_sqc = np.random.choice(list(path_dic.keys()), 2, replace=False)
    sqc1, sqc2 = path_dic[select_sqc[0]], path_dic[select_sqc[1]]
    select_list1, select_list2 = [], []
    if clip == 0:
        select_list1 = list(range(len(sqc1['img'])))
        select_list2 = list(range(len(sqc2['img'])))
    else:
        list1 = list(range(len(sqc1['img'])))
        list2 = list(range(len(sqc2['img'])))
        for begin in range(memory_size - 1, len(list1), clip):
            select_list1.append(list1[begin])
        for begin in range(memory_size - 1, len(list2), clip):
            select_list2.append(list2[begin])

    select_id1 = random.choice(select_list1)
    select_id2 = random.choice(select_list2)
    img_path1, gt_path1, bor_path1 = sqc1['img'][select_id1], sqc1['gt'][select_id1], sqc1['border'][select_id1]
    img_path2, gt_path2, bor_path2 = sqc2['img'][select_id2], sqc2['gt'][select_id2], sqc2['border'][select_id2]
    img1, gt1, bor1 = Image.open(img_path1).convert('RGB'), Image.open(gt_path1).convert("L"), Image.open(
        bor_path1).convert("L")
    img2, gt2, bor2 = Image.open(img_path2).convert('RGB'), Image.open(gt_path2).convert("L"), Image.open(
        bor_path2).convert("L")
    img1, gt1, bor1 = trans(img1, gt1, bor1)
    img2, gt2, bor2 = trans(img2, gt2, bor2)

    return img1, gt1, bor1, img2, gt2, bor2


def makebox(frames, location, borders, labels, epoch=0, idxx=0, config=None):
    # scale = random.uniform(0.8, 2.0)
    # longside = random.randint(int(base_size * 0.8), int(base_size * 2.0))
    # frames = F.interpolate(frames, size=(longside, longside), mode='bilinear', align_corners=False)
    # masks = F.interpolate(masks, size=(longside, longside), mode='bilinear', align_corners=True)
    bt, c, h, w = frames.size()

    total_over_box1, total_over_box2 = [], []
    total_crop_box1, total_crop_box2 = [], []
    sbor1, sbor2 = [], []
    synth1, synth2 = [], []
    total_label1, total_label2 = [], []
    total_bor1, total_bor2 = [], []

    for i in range(bt):
        bimg1, bgt1, bbor1, bimg2, bgt2, bbor2 = getbackground(config.sqc_path)

        arr_location = location[i].cpu().permute(1, 2, 0).numpy()
        arr_location = (arr_location * 255).astype(np.uint8)
        contours, hierarchy = cv2.findContours(arr_location, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]

        num = len(bounding_boxes)
        if num < 1:
            xp1 = random.randint(0, int(0.5 * base_size))
            yp1 = random.randint(0, int(0.5 * base_size))
            wp = random.randint(int(0.25 * base_size), int(0.5 * base_size))
            hp = random.randint(int(0.25 * base_size), int(0.5 * base_size))
        else:
            idx = random.randint(0, num - 1)
            [xp1, yp1, wp, hp] = bounding_boxes[idx]

        x1 = random.randint(max(0, int(xp1 + 0.75 * wp - crop)), int(xp1 + 0.25 * wp))  # 取样第一个bounding box的左上
        y1 = random.randint(max(0, int(yp1 + 0.75 * hp - crop)), int(yp1 + 0.25 * hp))
        x2 = random.randint(max(0, int(xp1 + 0.75 * wp - crop)), int(xp1 + 0.25 * wp))  # 取样第二个bounding box的左上角
        y2 = random.randint(max(0, int(yp1 + 0.75 * hp - crop)), int(yp1 + 0.25 * hp))
        # x2 = (x2 - x1) // stride * stride + x1
        # y2 = (y2 - y1) // stride * stride + y1

        if abs(x2 - x1) < stride:
            if x1 < x2: x2 += stride
            else:   x1 += stride
        if abs(y2 - y1) < stride:
            if y1 < y2: y2 += stride
            else:   y1 += stride

        if abs(crop - abs(x2 - x1)) < stride:
            if x1 < x2: x1 += stride
            else:   x2 += stride
        if abs(crop - abs(y2 - y1)) < stride:
            if y1 < y2: y1 += stride
            else:   y2 += stride

        ul1 = [max(0, y2 - y1), max(0, x2 - x1)]
        br1 = [min(crop, crop + y2 - y1), min(crop, crop + x2 - x1)]
        ul2 = [max(0, y1 - y2), max(0, x1 - x2)]
        br2 = [min(crop, crop + y1 - y2), min(crop, crop + x1 - x2)]

        try:
            assert (br1[0] - ul1[0]) * (br1[1] - ul1[1]) == (
                    br2[0] - ul2[0]) * (br2[1] - ul2[1])
        except:
            print("h: {}, w: {}".format(h, w))
            print("x1: {}, x2: {}, y1: {}, y2: {}".format(x1, x2, y1, y2))
            print("ul1: ", ul1)
            print("br1: ", br2)
            print("ul2: ", ul2)
            print("br2: ", br2)
            exit()

        image1 = frames[i, :, y1:min(y1 + crop, h), x1:min(x1 + crop, w)]
        image2 = frames[i, :, y2:min(y2 + crop, h), x2:min(x2 + crop, w)]
        label1 = labels[i, :, y1:min(y1 + crop, h), x1:min(x1 + crop, w)]
        label2 = labels[i, :, y2:min(y2 + crop, h), x2:min(x2 + crop, w)]
        border1 = borders[i, :, y1:min(y1 + crop, h), x1:min(x1 + crop, w)]
        border2 = borders[i, :, y2:min(y2 + crop, h), x2:min(x2 + crop, w)]

        _, crop_h1, crop_w1 = image1.shape
        _, crop_h2, crop_w2 = image2.shape

        pad1 = (0, max(0, crop - crop_w1), 0, max(0, crop - crop_h1))
        pad2 = (0, max(0, crop - crop_w2), 0, max(0, crop - crop_h2))
        image1 = F.pad(image1.unsqueeze(0), pad1).squeeze(0)
        label1 = F.pad(label1.unsqueeze(0), pad1).squeeze(0)
        border1 = F.pad(border1.unsqueeze(0), pad1).squeeze(0)
        image2 = F.pad(image2.unsqueeze(0), pad2).squeeze(0)
        label2 = F.pad(label2.unsqueeze(0), pad2).squeeze(0)
        border2 = F.pad(border2.unsqueeze(0), pad2).squeeze(0)

        syn1, [sx1, sy1], s_bor1 = synth(image1, bimg1, border1, bbor1)
        syn2, [sx2, sy2], s_bor2 = synth(image2, bimg2, border2, bbor2)

        crop_box1 = [i, sx1, sy1, crop + sx1, crop + sy1]
        crop_box2 = [i, sx2, sy2, crop + sx2, crop + sy2]

        over_box1 = [i, ul1[1], ul1[0], br1[1], br1[0]]
        over_box2 = [i, ul2[1], ul2[0], br2[1], br2[0]]
        if epoch < -1 and epoch % 5 == 0 and idxx % 10 and i == 0:
            dic = {
                'location': location[i],
                'frame': frames[i], 'label': labels[i],
                'img1': image1, 'img2': image2,
                'bor1': border1, 'bor2': border2,
                'back1': bimg1, 'back2': bimg2,
            }
            visualize(dic, config.visualize_path + '/cut', epoch, str(idxx) + '-' + str(i))
            dic = {  #
                's1': syn1, 's2': syn2,  
                'sb1': s_bor1, 'sb2': s_bor2,  
                'crop1': syn1[:, crop_box1[2]:crop_box1[4], crop_box1[1]:crop_box1[3]],
                'crop2': syn2[:, crop_box2[2]:crop_box2[4], crop_box2[1]:crop_box2[3]],
                'over1': image1[:, over_box1[2]:over_box1[4], over_box1[1]:over_box1[3]],
                'over2': image2[:, over_box2[2]:over_box2[4], over_box2[1]:over_box2[3]]

            }
            visualize(dic, config.visualize_path + '/cut', epoch, str(idxx) + '-' + str(i))


        total_over_box1.append(over_box1)
        total_over_box2.append(over_box2)
        total_crop_box1.append(crop_box1)
        total_crop_box2.append(crop_box2)
        sbor1.append(s_bor1)
        sbor2.append(s_bor2)
        synth1.append(syn1)
        synth2.append(syn2)
        total_label1.append(label1)
        total_label2.append(label2)
        total_bor1.append(border1)
        total_bor2.append(border2)

    dic1 = {
        'view': torch.stack(synth1, dim=0),  
        'label': torch.stack(total_label1, dim=0),  
        'bor': torch.stack(total_bor1, dim=0), 
        'sbor': torch.stack(sbor1, dim=0),  
        'overlap_box': total_over_box1,
        'crop_box': total_crop_box1,
    }
    dic2 = {
        'view': torch.stack(synth2, dim=0),
        'label': torch.stack(total_label2, dim=0),
        'bor': torch.stack(total_bor2, dim=0),
        'sbor': torch.stack(sbor2, dim=0),
        'overlap_box': total_over_box2,
        'crop_box': total_crop_box2,
    }
    return dic1, dic2


def synth(img, bimg, bor, bbor, label=None, blabel=None):
    c, h, w = img.shape
    bc, bh, bw = bimg.shape
    bimg1 = bimg.clone()
    bbor1 = bbor.clone()

    x = random.randint(int(bw * 0.2), int(bw * 0.8) - w)
    y = random.randint(int(bh * 0.2), int(bh * 0.8) - h)
    bimg1[:, y:y + h, x:x + w] = img[:, :, :]
    bbor1[:, y:y + h, x:x + w] = bor[:, :, :]
    syn = bbor1 * bimg1 + (1. - bbor1) * bimg
    if label is not None:
        blabel1 = blabel.clone()
        blabel1[:, y:y + h, x:x + w] = label[:, :, :]
        syn_label = bbor1 * blabel1 + (1. - bbor1) * blabel
        return syn, [x, y], bbor1, syn_label
    return syn, [x, y], bbor1


def makebox_supervise(frames, borders, labels, epoch=0, idxx=0, config=None):
    bt, c, h, w = frames.size()
    synth1, synth2 = [], []
    total_label1, total_label2 = [], []

    for i in range(bt):
        bimg1, bgt1, bbor1, bimg2, bgt2, bbor2 = getbackground(config.sqc_path)

        arr_location = labels[i].cpu().permute(1, 2, 0).numpy()
        arr_location = (arr_location * 255).astype(np.uint8)
        contours, hierarchy = cv2.findContours(arr_location, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]

        num = len(bounding_boxes)
        if num < 1:
            xp1 = random.randint(0, int(0.5 * base_size))
            yp1 = random.randint(0, int(0.5 * base_size))
            wp = random.randint(int(0.25 * base_size), int(0.5 * base_size))
            hp = random.randint(int(0.25 * base_size), int(0.5 * base_size))
        else:
            idx = random.randint(0, num - 1)
            [xp1, yp1, wp, hp] = bounding_boxes[idx]

        x1 = random.randint(max(0, int(xp1 + 0.75 * wp - crop)), int(xp1 + 0.25 * wp)) 
        y1 = random.randint(max(0, int(yp1 + 0.75 * hp - crop)), int(yp1 + 0.25 * hp))
        x2 = random.randint(max(0, int(xp1 + 0.75 * wp - crop)), int(xp1 + 0.25 * wp)) 
        y2 = random.randint(max(0, int(yp1 + 0.75 * hp - crop)), int(yp1 + 0.25 * hp))


        image1 = frames[i, :, y1:min(y1 + crop, h), x1:min(x1 + crop, w)]
        image2 = frames[i, :, y2:min(y2 + crop, h), x2:min(x2 + crop, w)]
        label1 = labels[i, :, y1:min(y1 + crop, h), x1:min(x1 + crop, w)]
        label2 = labels[i, :, y2:min(y2 + crop, h), x2:min(x2 + crop, w)]
        border1 = borders[i, :, y1:min(y1 + crop, h), x1:min(x1 + crop, w)]
        border2 = borders[i, :, y2:min(y2 + crop, h), x2:min(x2 + crop, w)]

        pad1 = (0, max(0, x1 + crop - w), 0, max(0, y1 + crop - h))  
        pad2 = (0, max(0, x2 + crop - w), 0, max(0, y2 + crop - h))
        image1 = F.pad(image1.unsqueeze(0), pad1).squeeze(0)
        label1 = F.pad(label1.unsqueeze(0), pad1).squeeze(0)
        border1 = F.pad(border1.unsqueeze(0), pad1).squeeze(0)
        image2 = F.pad(image2.unsqueeze(0), pad2).squeeze(0)
        label2 = F.pad(label2.unsqueeze(0), pad2).squeeze(0)
        border2 = F.pad(border2.unsqueeze(0), pad2).squeeze(0)

        syn1, [sx1, sy1], s_bor1, s_label1 = synth(image1, bimg1, border1, bbor1, label1, bgt1)
        syn2, [sx2, sy2], s_bor2, s_label2 = synth(image2, bimg2, border2, bbor2, label2, bgt2)

        if epoch < -1 and epoch % 5 == 0 and idxx % 10 and i == 0:
            dic = { 
                'frame': frames[i], 'label': labels[i],
                'img1': image1, 'img2': image2,
                'bor1': border1, 'bor2': border2,
                'back1': bimg1, 'back2': bimg2,
            }
            visualize(dic, config.visualize_path + '/cut', epoch, str(idxx) + '-' + str(i))
            dic = {  
                's1': syn1, 's2': syn2, 
                'sb1': s_bor1, 'sb2': s_bor2, 
                'sgt1': s_label1, 'sgt2': s_label2,
            }
            visualize(dic, config.visualize_path + '/cut', epoch, str(idxx) + '-' + str(i))

        synth1.append(syn1)
        synth2.append(syn2)
        total_label1.append(s_label1)
        total_label2.append(s_label2)

    dic1 = {
        'view': torch.stack(synth1, dim=0),  # 合成后得图像
        'label': torch.stack(total_label1, dim=0),  # crop之后的label
    }
    dic2 = {
        'view': torch.stack(synth2, dim=0),
        'label': torch.stack(total_label2, dim=0),
    }
    return dic1, dic2


if __name__ == '__main__':
    img_label_transform = Compose_imglabel([
        Resize1(base_size, base_size),
        toTensor(),
        Normalize11(statistics['mean'], statistics['std'])
    ])
    img_path = 'TrainSet/CVC-ClinicDB-612/1/Frame/26.bmp'
    gt_path = 'TrainSet/CVC-ClinicDB-612/1/GT/26.tif'
    bor_path = 'TrainSet/CVC-ClinicDB-612/1/border/26.bmp'
    frame = Image.open(img_path).convert('RGB')
    mask = Image.open(gt_path).convert('L')
    bor = Image.open(bor_path).convert('L')
    frame, mask, bor = img_label_transform(frame, mask, bor)
    frame = frame.unsqueeze(0).repeat(6, 1, 1, 1)
    mask = mask.unsqueeze(0).repeat(6, 1, 1, 1)
    bor = bor.unsqueeze(0).repeat(6, 1, 1, 1)
    # frame = frame.to(dtype=torch.float32)
    # mask = mask.to(dtype=torch.float32)
    for i in range(1000):
        print()
        makebox(frame, mask, bor, mask)
