import torch
import time
import os
import logging
import matplotlib.pyplot as plt

def get_logger(file_path):
    """ Make python logger """
    # [!] Since tensorboardX use default logger (e.g. logging.info()), we should use custom logger
    logger = logging.getLogger('unet')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%y-%m-%d %H:%M:%S %Z')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger


class Timer(object):
    def __init__(self):
        self.ticks = []
        self.record()
    
    def record(self):
        self.ticks.append(time.time())
    
    def elapsed_time(self, split=False, record=True):
        if record: self.record()
        if split: elapsed = self.ticks[-1] - self.ticks[-2]
        else: elapsed = self.ticks[-1] - self.ticks[0]
        return self.cal_time(elapsed)
    
    def _elapsed_split(self):
        elapsed = self.ticks[-1] - self.ticks[-2]
        return elapsed

    def cal_time(self, elapsed: float) -> str:    
        day = int(elapsed // (60*60*24)); elapsed %= (60*60*24)
        h = int(elapsed // (60*60)); elapsed %= (60*60) 
        m = int(elapsed // 60); elapsed %= 60
        s = int(elapsed)
        return f"{day:2d}day {h}h {m}min {s}sec"
    
    def eta(self, epoch: int, num_epochs: int) -> str:
        epochs_to_go = num_epochs - (epoch - 1)
        estimate = epochs_to_go * self._elapsed_split()
        return self.cal_time(estimate)


def mkdir(path):
    os.mkdir(path)


def check_dir(path):
    return os.path.exists(path)


def save(model, path: str, checkpoint: str):
    if not check_dir(path):
        mkdir(path)
    
    print("[*] Save Model...")
    torch.save(model.state_dict(), os.path.join(path, checkpoint))
    print("[*] Model Saved...")


def load(model, path: str):
    print("[*] Load Model...")
    model.load_state_dict(torch.load(path))
    print("[*] Model Loaded...")
    return model


def print_log(metrics: dict, epoch: int, total_epoch: int, step: int, total_step: int, phase: str = 'train', prnt=print):
    metrics_str = ' '.join(f' {metric}: {value:.5f}' for metric, value in metrics.items())
    prnt(f'[{phase.upper()}] Epoch {epoch}/{total_epoch} Step {step}/{total_step} {metrics_str}')


if __name__ == '__main__':
    timer = Timer()

    timer.record()
    time.sleep(10)
    timer.record()

    print(timer.elapsed_time())