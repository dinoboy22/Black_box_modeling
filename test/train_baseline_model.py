#!/usr/bin/env python
# coding: utf-8

import os
import sys
sys.path.append('./')
sys.path.append('../')

import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

import config
from libs.data_loader import BBDataModule
from libs.nn import BaselineModel

import numpy as np
import pandas as pd
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
    ap.add_argument('--wide', default=False, action=BooleanOptionalAction, help='select wide model')
    ap.add_argument('-f', '--file', help='train csv file')
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

    cfg = config.BASELINE_MODEL
    if args['wide']:
        cfg = config.BASELINE_WIDE_MODEL

    if args.get('file'):
        cfg['train_csv_file'] = args['file']

    ROOT_DIR = '.' if os.path.exists('config') else '..' 
    csv_file = os.path.join(ROOT_DIR, 'dataset', cfg['train_csv_file'])
    df = pd.read_csv(csv_file)
    np.testing.assert_equal(df.shape[1]-2, cfg['num_input'])

    model = BaselineModel(
        num_input=cfg['num_input'], 
        num_output=cfg['num_output'], 
        layers=cfg['layers'],
        dropout=cfg['dropout'],
        learning_rate=cfg['learning_rate']
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
    logger = TensorBoardLogger(log_dir, name=cfg['label'])

    checkpoint_dir = os.path.join(ROOT_DIR, 'models')
    checkpoint_filename = f'{cfg['label']}-v{logger.version}' + '-{epoch:02d}-{val_rmse:.2f}'
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=checkpoint_dir,
        filename=checkpoint_filename,
        save_top_k=3,
        mode='min',
    )

    trainer = pl.Trainer(
        # limit_train_batches=0.1, # use only 10% of the training data
        min_epochs=1,
        max_epochs=cfg['num_epochs'],
        precision='bf16-mixed',
        callbacks=[checkpoint_callback, EarlyStopping(patience=10, monitor="val_loss")],
        logger=logger,
        # profiler=profiler,
        # profiler='simple'
    )

    trainer.fit(model, data_module)
    trainer.validate(model, data_module)

    # # load the test datta (transformed)
    # test_csv = os.path.join(ROOT_DIR, 'dataset', 'test_pt.csv')
    # test_df = pd.read_csv(test_csv)
    # test_df.pop('ID')
    # test_df.pop('y')
    # X_test = torch.tensor(test_df.values).float()
    # model(X_test)
