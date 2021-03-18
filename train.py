import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF
from torch.utils.tensorboard import SummaryWriter

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp

from torchsummary import summary
from tqdm import tqdm

from model.unet import UNet
from dataloader import *
from config import *
from utils import *
from metrics import *
from transforms import *

best_miou = 0
NUM_OF_CLASS = 20

config = train_config()

if not check_dir(config.path):
    mkdir(config.path)

if not check_dir(os.path.join(config.path, config.name)):
    mkdir(os.path.join(config.path, config.name))

if not check_dir(os.path.join(config.path, config.name, 'checkpoints')):
    mkdir(os.path.join(config.path, config.name, 'checkpoints'))

logger = get_logger(os.path.join(config.path, config.name, "{}.log".format(config.name)))
writer = SummaryWriter(log_dir=os.path.join(config.path, config.name, "tb"))


def train(model, trainloader, optimizer, criterion, writer, logger, config, epoch, gpu):
    pixel_accs = AverageMeter()
    losses = AverageMeter()
    mIoUs = AverageMeter()
    metrics = {}
    
    model.train()
    for step, (x, y) in enumerate(trainloader):
        y = y - (y == 255).int() * 255
        # print(set(y.reshape(-1).tolist()))

        x = x.to(gpu)
        y = y.to(gpu)

        pred = model(x)
        loss = criterion(pred, y)

        losses.update(loss.item())
        mIoUs.update(mIoU(pred, y))
        pixel_accs.update(pixel_accuracy(pred, y))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (step+1) % config.print_freq == 0 or (step+1) == len(trainloader):
            metrics['loss'] = losses.avg
            metrics['mIoU'] = mIoUs.avg
            metrics['PixelAcc'] = pixel_accs.avg

            print_log(metrics, epoch=epoch, total_epoch=config.epoch, step=step+1, total_step=len(trainloader), phase='train', prnt=logger.info, gpu=gpu)      

    writer.add_scalar(f'Loss/gpu{gpu}/train', losses.avg, epoch)
    writer.add_scalar(f'mIoU/gpu{gpu}/train', mIoUs.avg, epoch)
    writer.add_scalar(f'PixelAcc/gpu{gpu}/train', pixel_accs.avg, epoch)

def evaluate(model, valloader, criterion, writer, logger, config, epoch, gpu):
    pixel_accs = AverageMeter()
    losses = AverageMeter()
    mIoUs = AverageMeter()
    metrics = {}

    model.eval()
    for step, (x, y) in enumerate(valloader):
        y = y - (y == 255).int() * 255

        x = x.to(gpu)
        y = y.to(gpu)

        pred = model(x)
        loss = criterion(pred, y)

        losses.update(loss.item())
        mIoUs.update(mIoU(pred, y))
        pixel_accs.update(pixel_accuracy(pred, y))

        if (step+1) % config.print_freq*2 == 0 or (step+1) == len(valloader):
            metrics['loss'] = losses.avg    
            metrics['mIoU'] = mIoUs.avg
            metrics['PixelAcc'] = pixel_accs.avg
            print_log(metrics, epoch=epoch, total_epoch=config.epoch, step=step+1, total_step=len(valloader), phase='val', prnt=logger.info, gpu=gpu)   

    writer.add_scalar(f'Loss/gpu{gpu}/val', losses.avg, epoch)
    writer.add_scalar(f'mIoU/gpu{gpu}/train', mIoUs.avg, epoch)
    writer.add_scalar(f'PixelAcc/gpu{gpu}/train', pixel_accs.avg, epoch)

    return mIoUs.avg, pixel_accs.avg

def main():    
    logger.info("Logger is set - training start")
    print_params(config, prtf=logger.info)

    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    torch.backends.cudnn.benchmark = True

    ngpus_per_node = len(config.gpus)

    config.world_size = ngpus_per_node * config.world_size
    mp.spawn(main_worker, nprocs=ngpus_per_node, 
             args=(ngpus_per_node, config))

def main_worker(gpu, ngpus_per_node, config):
    global best_miou
    timer = Timer()
    gpu = config.gpus[gpu]

    # set default gpu device id
    torch.cuda.set_device(gpu)
    print("Use GPU: {} for training".format(gpu))

    config.rank = config.rank * ngpus_per_node + gpu
    dist.init_process_group(backend='nccl', 
                            init_method='tcp://127.0.0.1:8080',
                            world_size=config.world_size, 
                            rank=config.rank)

    '''
    LOAD DATA
    '''
    train_transform = Compose([
        RandomHorizontalFlip(0.5),
        RandomResize(192, 768),
        RandomCrop(384),
        # ColorJitter(0.3),
        ToTensor(),
    ])

    val_transform = Compose([
        RandomResize(192, 768),
        CenterCrop(384),
        ToTensor(),
    ])

    # traindata = COCO_Dataset(split='train', transform=train_transform)
    # valdata = COCO_Dataset(split='validate', transform=val_transform)

    traindata = dset.VOCSegmentation(root='../datasets/img_type_datasets', 
                                     year='2012', 
                                     image_set='train', 
                                     transforms=train_transform)
    validdata = dset.VOCSegmentation(root='../datasets/img_type_datasets', 
                                     year='2012', 
                                     image_set='val', 
                                     transforms=val_transform)

    train_sampler = DistributedSampler(traindata)

    trainloader = DataLoader(traindata, 
                             batch_size=config.batch_size, 
                             sampler=train_sampler, 
                             shuffle=(train_sampler is None), 
                             num_workers=config.workers, 
                             drop_last=True,
                             pin_memory=True)
    validloader = DataLoader(validdata, 
                             batch_size=config.batch_size // 2, 
                             num_workers=config.workers, 
                             shuffle=False,
                             pin_memory=True)

    '''
    MODEL SETTING
    '''
    model = UNet(num_of_cls=NUM_OF_CLASS+1, use_upconv=False).to(gpu)
    if config.rank % ngpus_per_node == 0:
        summary(model, input_size=(3, 384, 384))
    model = DDP(model, device_ids=[gpu])

    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.init_lr, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.init_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-6, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epoch, eta_min=config.last_lr, verbose=True)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,70], gamma=0.1)

    '''
    TRAINING
    '''
    timer.record()
    for epoch in range(1, config.epoch+1):
        train_sampler.set_epoch(epoch)
        
        if config.rank % ngpus_per_node == 0:
            logger.info(f'Epoch {epoch} Learning Rate: {scheduler.get_lr()[0]}')
        
        dist.barrier()
        train(model, trainloader, optimizer, criterion, writer, logger, config, epoch, gpu)

        dist.barrier()
        miou, pixel_acc = evaluate(model, validloader, criterion, writer, logger, config, epoch, gpu)
        
        logger.info(f'GPU{gpu} Epoch Elapsed Time: {timer.elapsed_time(split=True)}')
        logger.info(f'GPU{gpu} ETA: {timer.eta(epoch, config.epoch)}')

        scheduler.step()
        
        if config.rank % ngpus_per_node == 0 and best_miou <= miou:
            best_miou = miou
            logger.info(f'GPU{gpu} Saving Best Checkpoint...')
            save(model, os.path.join(config.path, config.name, 'checkpoints'), 'best.pth')
            logger.info(f'Best Checkpoint Saved on GPU{gpu}')

    dist.barrier()
    if config.rank % ngpus_per_node == 0:
        logger.info(f'[*] Train Finished!!')
        logger.info(f'Total Elapsed Time: {timer.elapsed_time()}')


if __name__ == '__main__':
    main()
