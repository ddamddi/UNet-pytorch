import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF
from torch.utils.tensorboard import SummaryWriter

from torchsummary import summary

from model.unet import UNet
from dataloader import *
from config import *
from util import *
from metrics import *
from transforms import *

from tqdm import tqdm

device = torch.device('cuda')
best_miou = 0
NUM_OF_CLASS = 20

def train(model, trainloader, optimizer, criterion, writer, logger, args, epoch):
    losses = []
    mIoUs = []
    pixel_accs = []
    metrics = {}
    
    model.train()
    for step, (x, y) in enumerate(trainloader):
        y = y - (y == 255).int() * 255
        # print(set(y.reshape(-1).tolist()))

        x = x.to(device)
        y = y.to(device)

        pred = model(x)
        loss = criterion(pred, y)

        losses.append(loss.item())
        mIoUs.append(mIoU(pred, y))
        pixel_accs.append(pixel_accuracy(pred, y))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (step+1) % args.print_freq == 0 or (step+1) == len(trainloader):
            metrics['loss'] = np.mean(losses)
            metrics['mIoU'] = np.mean(mIoUs)
            metrics['PixelAcc'] = np.mean(pixel_accs)
            print_log(metrics, epoch=epoch, total_epoch=args.epoch, step=step+1, total_step=len(trainloader), phase='train', prnt=logger.info)      

    writer.add_scalar('Loss/train', np.mean(losses), epoch)
    writer.add_scalar('mIoU/train', np.mean(mIoUs), epoch)
    writer.add_scalar('PixelAcc/train', np.mean(pixel_accs), epoch)

def evaluate(model, valloader, criterion, writer, logger, args, epoch):
    losses = []
    mIoUs = []
    pixel_accs = []
    metrics = {}
    global best_miou

    model.eval()
    for step, (x, y) in enumerate(tqdm(valloader)):
        y = y - (y == 255).int() * 255

        x = x.to(device)
        y = y.to(device)

        pred = model(x)
        loss = criterion(pred, y)

        losses.append(loss.item())
        mIoUs.append(mIoU(pred, y))
        pixel_accs.append(pixel_accuracy(pred, y))
        
        # if (step+1) % args.print_freq == 0 or (step+1) == len(valloader):
    
    metrics['loss'] = np.mean(losses)
    metrics['mIoU'] = np.mean(mIoUs)
    metrics['PixelAcc'] = np.mean(pixel_accs)
    print_log(metrics, epoch=epoch, total_epoch=args.epoch, step=step+1, total_step=len(valloader), phase='val', prnt=logger.info)   

    writer.add_scalar('Loss/val', np.mean(losses), epoch)
    writer.add_scalar('mIoU/train', np.mean(mIoUs), epoch)
    writer.add_scalar('PixelAcc/train', np.mean(pixel_accs), epoch)

    logger.info(f'Saving Checkpoint {epoch}.pth...')
    save(model, os.path.join(args.path, args.name, 'checkpoints'), f'{epoch}.pth')
    logger.info(f'Checkpoint {epoch}.pth Saved')

    if best_miou <= metrics['mIoU']:
        logger.info('Saving Best Checkpoint...')
        save(model, os.path.join(args.path, args.name, 'checkpoints'), 'best.pth')
        logger.info(f'Best Checkpoint Saved')

        best_miou = metrics['mIoU']

def main():
    args = train_config()

    if not check_dir(args.path):
        mkdir(args.path)

    if not check_dir(os.path.join(args.path, args.name)):
        mkdir(os.path.join(args.path, args.name))

    if not check_dir(os.path.join(args.path, args.name, 'checkpoints')):
        mkdir(os.path.join(args.path, args.name, 'checkpoints'))
    
    logger = get_logger(os.path.join(args.path, args.name, "{}.log".format(args.name)))
    logger.info("Logger is set - training start")

    print_params(args, prtf=logger.info)

    writer = SummaryWriter(log_dir=os.path.join(args.path, args.name, "tb"))

    timer = Timer()

    # set default gpu device id
    torch.cuda.set_device(args.gpus[0])

    # set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True
    
    '''
    LOAD DATA
    '''
    train_transform = Compose([
        RandomHorizontalFlip(0.5),
        RandomResize(320, 1280),
        RandomCrop(640),
        # ColorJitter(0.3),
        ToTensor(),
    ])

    val_transform = Compose([
        RandomResize(320, 1280),
        CenterCrop(640),
        ToTensor(),
    ])

    # traindata = COCO_Dataset(split='train', transform=train_transform)
    # valdata = COCO_Dataset(split='validate', transform=val_transform)

    traindata = dset.VOCSegmentation(root='../datasets/img_type_datasets', year='2012', image_set='train', transforms=train_transform)
    valdata = dset.VOCSegmentation(root='../datasets/img_type_datasets', year='2012', image_set='val', transforms=val_transform)

    trainloader = DataLoader(traindata, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
    valloader = DataLoader(valdata, batch_size=1, num_workers=args.workers, shuffle=False)


    '''
    MODEL SETTING
    '''
    model = nn.DataParallel(UNet(num_of_cls=NUM_OF_CLASS+1, use_upconv=False), device_ids=args.gpus).to(device)
    summary(model, input_size=(3, 640, 640))

    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.init_lr, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-6, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch, eta_min=args.last_lr, verbose=True)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,70], gamma=0.1)

    '''
    TRAINING
    '''
    timer.record()
    for epoch in range(1, args.epoch+1):
        logger.info(f'Epoch {epoch} Learning Rate: {scheduler.get_lr()[0]}')
        train(model, trainloader, optimizer, criterion, writer, logger, args, epoch)
        evaluate(model, valloader, criterion, writer, logger, args, epoch)
        
        logger.info(f'Epoch Elapsed Time: {timer.elapsed_time(split=True)}')
        logger.info(f'ETA: {timer.eta(epoch, args.epoch)}')
        scheduler.step()
    
    save(model, os.path.join(args.path, args.name, 'checkpoints'), 'final.pth')
    writer.close()

    logger.info(f'[*] Train Finished!!')
    logger.info(f'Total Elapsed Time: {timer.elapsed_time()}')


if __name__ == '__main__':
    main()
