#!/usr/bin/env python
# coding: utf-8

import os
import sys
sys.path.append('./')
sys.path.append('../')

import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping

import config
from libs.data_loader import BBDataModule
from libs.nn import BaselineModel

# import numpy as np
# import pandas as pd
# from torch.utils.data import Dataset, DataLoader, random_split, default_collate

if __name__ == '__main__':
    # import freeze_support
    from multiprocessing import freeze_support
    freeze_support()

    cfg = config.BASELINE_MODEL

    ROOT_DIR = '.' if os.path.exists('config') else '..' 
    csv_file = os.path.join(ROOT_DIR, 'dataset', cfg['train_csv_file'])
    # csv_file = os.path.join(ROOT_DIR, 'dataset', 'train.csv')

    model = BaselineModel(
        num_input=cfg['num_input'], 
        num_output=cfg['num_output'], 
        layers=cfg['layers'],
        dropout=cfg['dropout']
    ) 
    # print(model)
    # testset = BBDataset(csv_file=csv_file, transform=None)
    # X, y = default_collate([testset[0]])
    # y_pred = model(X)

 
    data_module = BBDataModule(
        csv_file=csv_file, 
        batch_size=cfg['batch_size'], 
        num_workers=cfg['num_workers']
    )

    log_dir = os.path.join(ROOT_DIR, 'tb_logs')
    logger = TensorBoardLogger(log_dir, name="baseline")

    trainer = pl.Trainer(
        # limit_train_batches=0.1, # use only 10% of the training data
        min_epochs=1,
        max_epochs=cfg['num_epochs'],
        precision='bf16-mixed',
        callbacks=[EarlyStopping(monitor="val_loss")],
        logger=logger,
        # profiler=profiler,
        # profiler='simple'
    )

    trainer.fit(model, data_module)
    trainer.validate(model, data_module)

