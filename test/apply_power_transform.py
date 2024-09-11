#!/usr/bin/env python
# coding: utf-8

import os
import sys
sys.path.append('./')
sys.path.append('../')

import pandas as pd
from sklearn.preprocessing import PowerTransformer
import pickle
import matplotlib.pyplot as plt
import libs.plots as myplt

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
    ap.add_argument('--target', default=True, action=BooleanOptionalAction, help='include target column')
    ap.add_argument('-v', '--verbose', type=int, default=0, help='verbose level')
    ap.add_argument('--debug', default=False, action=BooleanOptionalAction, help='debug message')

    args = vars(ap.parse_args())
    if args['verbose']:
        logger.setLevel(logging.DEBUG)

    logger.info("Started...")
    logger.debug(f"Argument: {args}")

    logger.info(f"Load the train data")
    ROOT_DIR = '.' if os.path.exists('config') else '..' 
    train_csv = os.path.join(ROOT_DIR, 'dataset', 'train.csv')
    train_df = pd.read_csv(train_csv)

    logger.debug("Save the ID and y columns temporarily")
    train_id = train_df['ID']
    train_y = train_df['y'] 

    logger.debug(f"Opt out the ID and y columns (if necessary)")
    train_df.pop('ID')

    logger.info(f"Apply the power transformer to the data")
    pt = PowerTransformer()
    X_pt = pt.fit_transform(train_df.values)
    train_df_pt = pd.DataFrame(X_pt, columns=train_df.columns)
    logger.debug("lambda_ : ", pt.lambdas_)

    logger.info(f"Save the power transformer to power_transformer.pkl")
    pickle.dump(
        pt, 
        open(os.path.join(ROOT_DIR, 'dataset', 'power_transformer.pkl'), 'wb')
    )
    # test - load the power transformer
    # pt = pickle.load(open(os.path.join(ROOT_DIR, 'dataset', 'power_transformer.pkl'), 'rb'))

    logger.debug("Restore the ID")
    train_df.insert(loc=0, column='ID', value=train_id.values)
    train_df_pt.insert(loc=0, column='ID', value=train_id.values)
    # train_df['y'] = train_y

    logger.info(f"Save the power transformed data")

    if args['target']:
        logger.info("train_pt.csv was created")
        train_df_pt.to_csv(os.path.join(ROOT_DIR, 'dataset', 'train_pt.csv'), index=False)
    else:
        logger.info("train_pt_excl_y.csv was created")
        train_df_pt['y'] = train_y 
        train_df_pt.to_csv(os.path.join(ROOT_DIR, 'dataset', 'train_pt_excl_y.csv'), index=False)

    logger.info("Comapred the original and power transformed data")
    # columns to inspect
    # columns = train_df_pt.columns[1:] # except 'ID'
    columns = ['x_0', 'x_1', 'x_2', 'x_3', 'x_4', 'y']

    for col in columns:
        myplt.plot_feature_distribution(train_df, col)
        myplt.plot_feature_distribution(train_df_pt, col)

    plt.show()

