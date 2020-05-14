import argparse
import logging
import os

import torch
import torch.distributed as dist

from torch.nn import init

from ssd.engine.inference import do_evaluation
from ssd.config import cfg
from ssd.data.build import make_data_loader
from ssd.engine.trainer import do_train
from ssd.modeling.detector import build_detection_model
from ssd.solver.build import make_optimizer, make_lr_scheduler
from ssd.utils import dist_util, mkdir
from ssd.utils.checkpoint import CheckPointer
from ssd.utils.dist_util import synchronize
from ssd.utils.logger import setup_logger
from ssd.utils.misc import str2bool


def train(cfg):
    logger = logging.getLogger('SSD.trainer')
    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)
    return model


def main():

    torch.backends.cudnn.benchmark = True
    cfg.merge_from_file('configs/vgg_ssd300_voc0712.yaml')
    cfg.freeze()

    model = train(cfg)
    model = model.eval()
    
    traced_script_module = torch.jit.script(model)
    #traced_script_module.save('new.pt')


if __name__ == '__main__':
    main()
