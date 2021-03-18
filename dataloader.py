import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as dset

from pycocotools.coco import COCO
from pycocotools import mask
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from PIL import Image

from transforms import Compose, RandomCrop, ToTensor

class COCO_Dataset(Dataset):
    def __init__(self, split='val', transform=None):
        super(COCO_Dataset, self).__init__()
        assert split in ['validate', 'train', 'test'], 'split argument should be train, validate or test'
        self.split = split

        self.data_dir = '../datasets/img_type_datasets/cocodataset/2017'
        self.transform = transform

        if self.split == 'train':
            self.data_type = 'train2017'
            annFile = '{}/annotations/instances_train2017.json'.format(self.data_dir)
        
        if self.split == 'validate':
            self.data_type = 'val2017'
            annFile = '{}/annotations/instances_val2017.json'.format(self.data_dir)
        
        if self.split == 'test':
            self.data_type = 'test2017'
            annFile = '{}/annotations/image_info_test2017.json'.format(self.data_dir)

        self.coco = COCO(annFile)

        cats = self.coco.loadCats(self.coco.getCatIds())

        self.cat_names = ['background']
        self.cat_names.extend(cat['name'] for cat in cats)

        self.cat_ids = [0]
        self.cat_ids.extend(cat['id'] for cat in cats)

        self.anno_img_id = []
        self.no_anno_img_id = []
        

        if self.split != 'test':
            for idx in self.coco.getImgIds():
                anno_ids = self.coco.getAnnIds(imgIds=idx, iscrowd=False)
                if len(anno_ids) == 0:
                    self.no_anno_img_id.append(idx)
                else:
                    self.anno_img_id.append(idx)
        else:
            self.anno_img_id = self.coco.getImgIds()

    def __getitem__(self, idx):
        img = self._load_img(self.anno_img_id[idx])
        w, h = img.size

        if self.split == 'test':
            seg_mask = np.zeros((w, h, 1))
        else:
            seg_mask = self._gen_seg_mask(self.anno_img_id[idx], h, w)
        
        if self.transform:
            img, seg_mask = self.transform(img, seg_mask)

        return img, seg_mask

    def __len__(self):
        return len(self.anno_img_id)
    
    def _load_img(self, img_id):
        img_info = self.coco.loadImgs(img_id)[0]
        img = Image.open(os.path.join(self.data_dir, self.data_type, img_info['file_name']))
        
        if img.mode != 'RGB':
            img = img.convert('RGB')

        return img

    def _gen_seg_mask(self, img_id, h, w):
        anno_info = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))

        mask_map = np.zeros((h,w), dtype=np.uint8)
        for anno in anno_info:
            rle = mask.frPyObjects(anno['segmentation'], h, w)
            m = mask.decode(rle)
            cat = anno['category_id']
            
            if cat in self.cat_ids:
                c = self.cat_ids.index(cat)
            else:
                continue

            if len(m.shape) < 3:
                mask_map[:, :] += (mask_map == 0) * (m * c)
            else:
                mask_map[:, :] += (mask_map == 0) * (((np.sum(m, axis=2)) > 0) * c).astype(np.uint8)

        return Image.fromarray(mask_map)


if __name__ == '__main__':

    transform = Compose([
        # RandomCrop(256),
        ToTensor()
    ])

    voc2012 = dset.VOCSegmentation(root='../datasets/img_type_datasets', year='2012', image_set='train', transforms=transform)
    dl = DataLoader(voc2012, batch_size=1, shuffle=True, drop_last=True)
    

    # ds = COCO_Dataset(split='validate', transform=transform)
    # dl = DataLoader(ds, batch_size=128, shuffle=True, drop_last=True)
    
    # print(ds.cat_ids)
    # print(len(ds.cat_ids))
    dl = iter(dl)
    h = []
    w = []
    rgb = [[],[],[]]
    for idx in range(len(voc2012)):
    #     # if idx < 648 and idx > 655:
    #         # pass

        x,y = next(dl)
        print(x.shape)
        h.append(x.size(2))
        w.append(x.size(3))

        x = np.asarray(x)[0]
        for i in range(3):
            rgb[i].append(x[i].mean())
        #print(x.shape)
        #print(y.shape)
        # print(y)
        # y = y.reshape(-1).tolist()
        # print(set(y))

    #     # r = random.randint(0, 0)

    #     # plt.imshow(x[r].permute(1,2,0))
    #     # plt.axis('off')
    #     # plt.show()

        # plt.imshow(y[0])
        # plt.axis('off')
        # plt.show()

    # for x,y in dl:
    #     print(x.shape)
    #     print(y.shape)
    # print("Fin.")
    w = sorted(w)
    h = sorted(h)

    print(np.mean(rgb, axis=1))

    print("W")
    print(w[-1], w[0])
    print("H")
    print(h[-1], h[0])

    print(len(voc2012))
