"""
XAI methods analyse model.
"""
import os
import numpy as np
from os.path import join
import cv2
import random
import datetime
import time
import yaml
import pickle
from tqdm import tqdm
from copy import deepcopy
from PIL import Image as pil_image
from metrics.utils import get_test_metrics
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
import matplotlib.pyplot as plt


from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from dataset.ff_blend import FFBlendDataset
from dataset.fwa_blend import FWABlendDataset
from dataset.pair_dataset import pairDataset

from trainer.trainer import Trainer
from detectors import DETECTOR
from metrics.base_metrics_class import Recorder
from collections import defaultdict

import argparse
from logger import create_logger

from XAI.utils.method_loader import load_explainer

# ===== set arguments =====
parser = argparse.ArgumentParser(description='XAI Analysis.')
parser.add_argument('--detector_path', type=str, 
                    default='./training/config/detector/xception.yaml',
                    help='path to detector YAML file')
parser.add_argument('--test_dataset', nargs='+', required=True,
                    help='One or more dataset names to test')
parser.add_argument('--weights_path', type=str, 
                    default='./training/weights/xception_best.pth')
parser.add_argument('--methods', nargs='+', required=True,
                    help='One or more explainability methods, e.g. gradcam ig lrp')
parser.add_argument('--output_dir', type=str,
                    default='./training/XAI/output/',
                    help='Directory to save gradient maps')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== set seed =====
def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    torch.manual_seed(config['manualSeed'])
    if config['cuda']:
        torch.cuda.manual_seed_all(config['manualSeed'])

# ===== Data loading =====
def prepare_testing_data(config):
    def get_test_data_loader(config, test_name):
        # update the config dictionary with the specific testing dataset
        config = config.copy()  # create a copy of config to avoid altering the original one
        config['test_dataset'] = test_name  # specify the current test dataset
        test_set = DeepfakeAbstractBaseDataset(config=config, mode='test')
        test_data_loader = torch.utils.data.DataLoader(
                dataset=test_set, 
                batch_size=config['test_batchSize'],
                shuffle=False, 
                num_workers=int(config['workers']),
                collate_fn=test_set.collate_fn,
                drop_last=False
            )
        return test_data_loader

    test_data_loaders = {}
    for one_test_name in config['test_dataset']:
        test_data_loaders[one_test_name] = get_test_data_loader(config, one_test_name)
    return test_data_loaders
    
def replace_relu(module):
    for name, child in module.named_children():
        if isinstance(child, torch.nn.ReLU):
            setattr(module, name, torch.nn.ReLU(inplace=False))
        else:
            replace_relu(child)

def main():
    # load config
    with open(args.detector_path, 'r') as f:
        config = yaml.safe_load(f)
    with open('./training/config/test_config.yaml', 'r') as f:
        config2 = yaml.safe_load(f)
    config.update(config2)

    config['test_dataset'] = args.test_dataset
    config['weights_path'] = args.weights_path
    config['output_dir'] = args.output_dir

    weights_path = args.weights_path
    methods = args.methods  

    # init seed
    init_seed(config)

    # set cudnn benchmark if needed
    if config.get('cudnn', False):
        cudnn.benchmark = True

    # prepare the testing data loader
    test_data_loaders = prepare_testing_data(config)
    print('===> Successfully load data.')

    # prepare the model (detector)
    model_class = DETECTOR[config['model_name']]
    model = model_class(config).to(device)
    ckpt = torch.load(weights_path, map_location=device)
    model.load_state_dict(ckpt, strict=True)
    model.eval()
    print('===> Successfully load model.')

    # ============ run XAI for each method ============
    for method in methods:
        # use a per-method copy of config to avoid side effects
        config_m = config.copy()
        config_m['method'] = method

        # prepare explainability method
        explainer = load_explainer(method, model, config_m)
        print(f'===> Successfully load XAI method: {method}')

        # start testing
        for dataset_name, loader in test_data_loaders.items():
            print(f"\n analysing dataset: {dataset_name} | method: {method}")

            output_dir = os.path.join(args.output_dir, method, dataset_name)
            os.makedirs(output_dir, exist_ok=True)

            START_AT = 0  # global image index to resume from

            # cache image paths once
            image_paths_all = loader.dataset.data_dict['image']

            for i, data_dict in tqdm(enumerate(loader), total=len(loader)):
                data_dict['dataset_name'] = dataset_name

                # add image_paths
                bs = data_dict['image'].size(0)
                start_idx = i * bs

                if start_idx < START_AT:
                    continue

                data_dict['image_paths'] = image_paths_all[start_idx:start_idx + bs]

                # move tensors to device
                for k in data_dict:
                    if isinstance(data_dict[k], torch.Tensor):
                        data_dict[k] = data_dict[k].to(device)

                # call xai method
                explanation = explainer.generate(data_dict)  # [B, 3, H, W]

            print(f"Analyse done: {dataset_name} | method: {method}\nSave in: {output_dir}.")     

if __name__ == '__main__':
    main()