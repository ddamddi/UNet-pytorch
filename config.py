import argparse
import torch


def parse_gpus(gpus):
    if gpus == 'all':
        return list(range(torch.cuda.device_count()))
    else:
        return [int(s) for s in gpus.split(',')]

def print_params(args: argparse.Namespace, prtf=print):
    prtf("")
    prtf("Parameters:")
    for attr, value in sorted(vars(args).items()):
        prtf("{}={}".format(attr.upper(), value))
    prtf("")


def train_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=str, default='all')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--init_lr', type=float, default=5e-4)
    parser.add_argument('--last_lr', type=float, default=0)
    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--print_freq', type=int, default=50)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--path', type=str, default='./results')
    parser.add_argument('--name', type=str, default='unet')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)

 
    args = parser.parse_args()

    args.gpus = parse_gpus(args.gpus)
    
    return args

def predict_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_dir', type=str, default='./checkpoints/')
    parser.add_argument('--path', type=str, default='./results')
    parser.add_argument('--name', type=str, default='unet')
    parser.add_argument('--img_path', type=str, default=None)
    parser.add_argument('--eval', action='store_true')

    args = parser.parse_args()
    
    return args

if __name__ == '__main__':
    pass