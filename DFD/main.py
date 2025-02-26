import os


import torch
import numpy as np
import random
import argparse
import importlib.util

from DIRG.dataset.dirg_dataset import DIRGDataloader

# load the config files
parser = argparse.ArgumentParser(description='Choose the configs to run.')
parser.add_argument('-c', '--config', type=str, required=True)
args = parser.parse_args()

use_config_spec = importlib.util.spec_from_file_location(
    args.config, "configs/{}.py".format(args.config))
config_module = importlib.util.module_from_spec(use_config_spec)
use_config_spec.loader.exec_module(config_module)
opt = config_module.opt

# set which gpu to use
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_device

# random seed specification
np.random.seed(opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)

# init model
from model.model import DFD as Model



model = Model(opt).to(opt.device)

print(model)

dataloader = DIRGDataloader(opt)
model.__log_write__(f'Source domain:{opt.src_domain_idx},target domain:{opt.tgt_domain_idx}')
# train
for epoch in range(opt.num_epoch):
    model.learn(epoch, dataloader)
    if (epoch + 1) % opt.save_interval == 0 or (epoch + 1) == opt.num_epoch:
        model.save()
    if (epoch + 1) % opt.test_interval == 0 or (epoch + 1) == opt.num_epoch:
        model.test(epoch, dataloader)
