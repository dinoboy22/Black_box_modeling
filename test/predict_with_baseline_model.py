#!/usr/bin/env python
# coding: utf-8

import os
import sys
sys.path.append('./')
sys.path.append('../')

import numpy as np
import pandas as pd
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping
import pickle

import config
from libs.data_loader import BBDataModule
from libs.nn import BaselineModel
# from torch.utils.data import Dataset, DataLoader, random_split, default_collate

if __name__ == '__main__':
    ## logger 셋팅
    import logging
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s %(levelname)s \t %(message)s")
    logger = logging.getLogger(__name__)

    ## CLI 셋팅
    import argparse
    from argparse import BooleanOptionalAction
    ap = argparse.ArgumentParser()
    ap.add_argument('-f', '--file', default='test.csv', help='test csv file')
    ap.add_argument('-v', '--verbose', type=int, default=0, help='verbose level')
    ap.add_argument('--debug', default=False, action=BooleanOptionalAction, help='debug message')

    args = vars(ap.parse_args())
    if args['verbose']:
        logger.setLevel(logging.DEBUG)

    logger.info("Started...")
    logger.debug(f"Argument: {args}")

    # import freeze_support
    from multiprocessing import freeze_support
    # freeze_support()

    # cfg = config.BASELINE_MODEL

    ROOT_DIR = '.' if os.path.exists('config') else '..' 
    test_csv_file = os.path.join(ROOT_DIR, 'dataset', 'test.csv')
    test_df = pd.read_csv(test_csv_file)
    test_df.pop('ID')
    test_df.pop('y')

    pt_file = os.path.join(ROOT_DIR, 'dataset', 'power_transformer.pkl')
    with open(pt_file, 'rb') as f:
        pt = pickle.load(f)

    X_test = pt.transform(test_df.values)








    # csv_file = os.path.join(ROOT_DIR, 'dataset', 'train.csv')

    # model = BaselineModel(
    #     num_input=cfg['num_input'], 
    #     num_output=cfg['num_output'], 
    #     layers=cfg['layers'],
    #     dropout=cfg['dropout']
    # ) 
 
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

    # load the model
    checkpoint = os.path.join(ROOT_DIR, 'models', 'baseline_model.ckpt')
    model = BaselineModel.load_from_checkpoint(
        checkpoint,
        num_input=cfg['num_input'],
        num_output=cfg['num_output'],
        layers=cfg['layers'],
        dropout=cfg['dropout']
    )
    trainer.validate(model, data_module)
    trainer.test(model, data_module)



test_csv = os.path.join(ROOT_DIR, 'dataset', 'test.csv')
# test_csv = os.path.join(ROOT_DIR, 'dataset', 'test_pt.csv')
test_df = pd.read_csv(test_csv)
X_test = torch.tensor(test_df.values).float()
X_test = torch.tensor(test_df.values, dtype=torch.float32)
        
