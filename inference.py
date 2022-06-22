import argparse
import os
import sys
import yaml

import torch
from torch.utils.data.dataloader import DataLoader

import numpy as np
from scipy.stats import norm
from sklearn import metrics

from model import create_gru_autoencoder
from data_loader import TimeSeriesDataset
from utils import get_config_from_yaml, save_config


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda" and not torch.cuda.is_initialized():
    torch.cuda.init()

def load_model(config):
    model = create_gru_autoencoder(d_model=config['d_model'],
     embed_dim=config['embed_dim'], n_layers=config['n_layers'], 
     l_win=config['l_win'], batch_size=config['batch_size'])

    model.load_state_dict(torch.load(
        config['checkpoint_dir'] + 'best_model.pt'))

    model.float().eval()
    return model

def create_labels(idx_anomaly_test, n_sample_test, config):
    anomaly_index = []
    test_labels = np.zeros(n_sample_test)
    for i in range(len(idx_anomaly_test)):
        idx_start = idx_anomaly_test[i] - config['l_win'] + 1
