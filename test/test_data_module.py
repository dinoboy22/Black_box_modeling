#!/usr/bin/env python
# coding: utf-8

import os
import sys
sys.path.append('./')
sys.path.append('../')

import numpy as np
import pandas as pd
import config
from libs.data_loader import BBDataset, BBDataModule


if __name__ == '__main__':
    # import freeze_support
    from multiprocessing import freeze_support
    freeze_support()


    ROOT_DIR = '.' if os.path.exists('config') else '..' 
    csv_file = os.path.join(ROOT_DIR, 'dataset', 'train.csv')
    # train_df = pd.read_csv(train_csv)
    # train_df = train_df.drop(columns=['ID'])

    ## BBDataset instance 
    dataset = BBDataset(csv_file)
    print(dataset.df.head())

    ## BBDataset#__len__() method test
    print(len(dataset))

    ## BBDataset#__getitem__() method test
    print(dataset[0])
    print(dataset[100])


    ## BBDataModule instance
    data_dir = os.path.join(ROOT_DIR, 'dataset')
    data_module = BBDataModule(csv_file=csv_file, batch_size=32, num_workers=4)
    print(type(data_module))

    ## BBDataModule#setup() method test
    data_module.setup()
    print(data_module.train_ds)
    print(data_module.val_ds)
    print(data_module.test_ds)

    print(len(data_module.train_ds))
    print(len(data_module.val_ds))
    print(len(data_module.test_ds))

    ## BBDataModule#train_dataloader() method test
    train_loader = data_module.train_dataloader()
    print(type(train_loader))
    for X, y in train_loader:
        print(X.shape, y.shape)

 
