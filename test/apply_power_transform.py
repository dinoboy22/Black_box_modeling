#!/usr/bin/env python
# coding: utf-8

import os
import sys
sys.path.append('./')
sys.path.append('../')

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PowerTransformer
import libs.plots as myplt
import config

# import numpy as np
# from scipy import stats
# import torchvision.transforms as transforms

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
    ap.add_argument('-f', '--file', default='train.csv', help='select train csv file')
    ap.add_argument('--target', default=True, action=BooleanOptionalAction, help='include target column')
    ap.add_argument('-v', '--verbose', type=int, default=0, help='verbose level')
    ap.add_argument('--debug', default=False, action=BooleanOptionalAction, help='debug message')

    args = vars(ap.parse_args())
    if args['verbose']:
        logger.setLevel(logging.DEBUG)

    logger.info("Started...")
    logger.debug(f"Argument: {args}")

    ROOT_DIR = '.' if os.path.exists('config') else '..' 
    train_csv = os.path.join(ROOT_DIR, 'dataset', args['file'])
    train_df = pd.read_csv(train_csv)

    train_id = train_df['ID']
    train_y = train_df['y'] 

    train_df.pop('ID')
    if not args['target']:
        train_df.pop('y')

    pt = PowerTransformer()
    X_pt = pt.fit_transform(train_df.values)
    train_df_pt = pd.DataFrame(X_pt, columns=train_df.columns)
    # pt_lambdas = pd.DataFrame({'cols':train_df.columns , 'pt_lambdas': pt.lambdas_})

    # save the power transformed data
    train_df.insert(loc=0, column='ID', value=train_id.values)
    train_df_pt.insert(loc=0, column='ID', value=train_id.values)

    if args['target']:
        train_df_pt.to_csv(os.path.join(ROOT_DIR, 'dataset', 'train_pt.csv'), index=False)
        print('Saved to train_pt.csv')
    else:
        train_df['y'] = train_y 
        train_df_pt['y'] = train_y 
        train_df_pt.to_csv(os.path.join(ROOT_DIR, 'dataset', 'train_pt_excl_y.csv'), index=False)
        print('Saved to train_pt_excl_y.csv')

    # columns to inspect
    # columns = train_df_pt.columns[1:] # except 'ID'
    columns = ['x_0', 'x_1', 'x_2', 'x_3', 'x_4', 'y']

    for col in columns:
        myplt.plot_feature_distribution(train_df, col)
        myplt.plot_feature_distribution(train_df_pt, col)

    plt.show()


