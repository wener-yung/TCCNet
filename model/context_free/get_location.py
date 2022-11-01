import torch
import numpy as np
import torch.nn as nn
import cv2


def get_location_dilate(pred_seg, pred_prop, first_mask, device):
    b, t, c, h, w = pred_seg.shape
    pred_seg = pred_seg.view(b * t, c, h, w)
    new_prop = torch.cat([first_mask.unsqueeze(1), pred_prop], dim=1)
    pred_prop = new_prop.view(b * t, c, h, w)
    location = (pred_seg + pred_prop) / 2
    pseudo_label = (pred_prop > 0.5).float().detach()

    location_thre = (location > 0.5).float()
    location_thre = location_thre.cpu().detach().numpy().astype(np.uint8)
    kernel_e = np.ones((5, 5), np.uint8)
    kernel_d = np.ones((10, 10), np.uint8)


    locas = []
    for i in range(location_thre.shape[0]):
        loca = location_thre[i]
        loca = cv2.erode(loca, kernel_e, iterations=1)
        loca = cv2.dilate(loca, kernel_d, iterations=1)
        loca = torch.tensor(loca, dtype=torch.float32, requires_grad=True).to(device)
        locas.append(loca)

    return torch.stack(locas, dim=0).detach(), pseudo_label

def get_location(pred_seg, pred_prop, first_mask, device):
    """
    Args:
        2,3,1,h,w,;; 2,2,1,h,w,;; 2,1,1,h,w
    Returns:
    """
    b, t, c, h, w = pred_seg.shape
    pred_seg = pred_seg.view(b*t, c, h, w)
    new_prop = torch.cat([first_mask.unsqueeze(1), pred_prop], dim=1)
    pred_prop = new_prop.view(b*t, c, h, w)
    location = (pred_seg + pred_prop) / 2

    pseudo_label = (pred_prop > 0.5).float()

    return location.detach(), pseudo_label.detach()



