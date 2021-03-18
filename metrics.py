import torch
import torch.nn
import torch.nn.functional as F

import numpy as np

def pixel_accuracy(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> float:
    with torch.no_grad():
        pred_mask = F.softmax(pred_mask, dim=1)
        pred_mask = torch.argmax(pred_mask, dim=1)
        
        correct = (pred_mask == gt_mask).int()
        accuracy = float(correct.sum()) / float(correct.numel())
        
    return accuracy


def mIoU(pred_mask: torch.Tensor, gt_mask: torch.Tensor, smooth=1e-10) -> float:
    with torch.no_grad():
        num_classes = pred_mask.size(1)

        pred_mask = F.softmax(pred_mask, dim=1)
        pred_mask = torch.argmax(pred_mask, dim=1)
        pred_mask = pred_mask.contiguous().view(-1)
        gt_mask = gt_mask.contiguous().view(-1)

        iou_per_class = []
        for clss in range(num_classes):
            true_class = (pred_mask.int() == clss)
            true_label = (gt_mask.int() == clss)

            if true_label.int().sum().item() == 0:
                iou_per_class.append(np.nan)
            else:
                I = torch.logical_and(true_class, true_label).sum().float().item()
                U = torch.logical_or(true_class, true_label).sum().float().item()

                iou = (I + smooth) / (U + smooth)
                iou_per_class.append(iou)
        
    return np.nanmean(iou_per_class)


def AP(iou: float) -> float:
    return 0.

def mAP() -> float:
    AP_per_iou = []
    for i in range(0.5, 1, 0.05):
        AP_per_iou = AP(i/100)
    
    return np.mean(AP_per_iou)


if __name__ == '__main__':
    a = torch.randn(1, 81, 4, 4)
    b = torch.zeros(1, 1, 4, 4)
    b[0,0,1,1] = 1
    b[0,0,1,2] = 1
    print(pixel_accuracy(a, b))
