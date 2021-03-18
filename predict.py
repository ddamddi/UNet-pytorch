import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np

from model.unet import UNet
from config import *
from utils import *
from visualize import *
from dataloader import *
from transforms import ToTensor, Compose, RandomResize, CenterCrop

from torchvision import transforms
from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# TODO: Segmentation Mask Prediction 구현
def predict(model, img=None, num_of_samples=20):
    # model.to(device)
    model.eval()

    val_transform = Compose([
        # RandomResize(256, 512),
        # CenterCrop(256),
        ToTensor(),
    ])

    if img == None:
        num_classes = 20 + 1
        # testdata = COCO_Dataset(split='validate', transform=val_transform)
        testdata = dset.VOCSegmentation(root='../../datasets', year='2012', image_set='val', transforms=val_transform)	
        testloader = DataLoader(testdata, batch_size=1, num_workers=1, shuffle=False)
        it = iter(testloader)
        for _ in range(num_of_samples):
            x, y = next(it)
            x = x.to(device)
            y = y - (y == 255).int() * 255

            pred = model(x)
            pred = pred.detach().cpu()

            mask = F.softmax(pred, dim=1)
            mask = torch.argmax(mask, dim=1)
            mask = np.asarray(mask.permute(1,2,0))
            
            img = np.asarray(x.detach().cpu().squeeze().permute(1,2,0) * 255)
            gt = np.asarray(y.permute(1,2,0))
            
            color_map = random_colors(num_classes, bright=True)
            mask = color_mask(mask, num_classes, color_map)
            gt = color_mask(gt, num_classes, color_map)
            	
            display_images([img, mask, gt])
    
    
    else:
        o_img = Image.open(img)
        img = transforms.ToTensor()(o_img)
        # img = torch.squeeze(img)

        # img = img.view(1, *img.permute(1,2,0))
        # img = img.permute(1,2,0)
        input_img = img.reshape(1, *img.size())
        input_img = input_img.to(device)
        print(input_img.shape)

        pred = model(input_img)
        pred = pred.detach().cpu()
        print(pred.shape)
        mask = F.softmax(pred, dim=1)
        mask = torch.argmax(mask, dim=1)
        print(mask.shape)
        
        mask = mask.permute(1,2,0)
        # img = img.permute(1,2,0)

        mask = np.array(mask)
        imask = np.stack([mask]*3, axis=2).squeeze()
        print(mask.shape)
        print(mask.shape[:-1])
        # img = Image.fromarray(np.array(img))

        o_img = np.array(o_img)

        display_images([o_img, color_mask(imask, 81), mask])


def evaluate(model):
    eval_transform = Compose([
        RandomResize(384, 384),
        CenterCrop(384),
        ToTensor(),
    ])

    testdata = dset.VOCSegmentation(root='../../datasets', year='2012', image_set='val', transforms=eval_transform)
    testloader = DataLoader(testdata, batch_size=1, num_workers=1, shuffle=True)

    mIoUs = []
    pixel_accs = []
    model.eval()
    for step, (x, y) in tqdm(enumerate(testloader)):
        x = x.to(device)
        y = y.to(device)
        y = y - (y == 255).int() * 255

        pred = model(x)
        mIoUs.append(mIoU(pred, y))
        pixel_accs.append(pixel_accuracy(pred, y))

    mean_mIoU = np.mean(mIoUs)
    mean_pixelAcc = np.mean(pixel_accs)
    
    print(f"mean_mIoU : {mean_mIoU}")
    print(f"m_pixelAcc : {mean_pixelAcc}")


if __name__ == '__main__':
    args = predict_config()

    model = UNet(num_of_cls=21, use_upconv=False).cuda(device)
    state_dict = torch.load(os.path.join(args.path, args.name, 'checkpoints', 'best.pth'), map_location='cuda:0')    

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    # print(state)
    # model = load(model, os.path.join(args.path, args.name, 'checkpoints', 'best.pth'))

    if args.eval:
        evaluate(model)
    else:
        predict(model, args.img_path)