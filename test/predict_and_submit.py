#!/usr/bin/env python
# coding: utf-8

import os
import sys
sys.path.append('./')
sys.path.append('../')

import numpy as np
import pandas as pd
import torch
import pickle

import config
from libs.nn import BaselineModel

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
    ap.add_argument('--target', default=True, action=BooleanOptionalAction, help='include target column')
    ap.add_argument('-v', '--verbose', type=int, default=0, help='verbose level')
    ap.add_argument('--debug', default=False, action=BooleanOptionalAction, help='debug message')

    args = vars(ap.parse_args())
    if args['verbose']:
        logger.setLevel(logging.DEBUG)

    logger.info("Started...")
    logger.debug(f"Argument: {args}")

    # # import freeze_support
    # from multiprocessing import freeze_support
    # freeze_support()

    logger.info(f"Select model configuration")
    cfg = config.BASELINE_MODEL
    if args['wide']:
        cfg = config.BASELINE_WIDE_MODEL

    ROOT_DIR = '.' if os.path.exists('config') else '..' 

    logger.info(f"Load the test data")
    test_csv_file = os.path.join(ROOT_DIR, 'dataset', 'test.csv')
    test_df = pd.read_csv(test_csv_file)
    test_id = test_df.pop('ID')
    test_df['y'] = np.ones_like(test_id, dtype=np.float32) # dummy y

    logger.info(f"Load the power transformer")
    pt_file = os.path.join(ROOT_DIR, 'dataset', 'power_transformer.pkl')
    with open(pt_file, 'rb') as f:
        pt = pickle.load(f)

    logger.info(f"Apply the power transformer to the test data")
    X_test = pt.transform(test_df.values)
    X_test = X_test[:, :-1] # remove the dummy y

    logger.info(f"Load the model and make the prediction")
    checkpoint = os.path.join(ROOT_DIR, 'models', 'baseline_model.ckpt')
    model = BaselineModel.load_from_checkpoint(
        checkpoint,
        num_input=cfg['num_input'],
        num_output=cfg['num_output'],
        layers=cfg['layers'],
        dropout=cfg['dropout'],
        learning_rate=cfg['learning_rate']
    )
    X_test = torch.tensor(X_test, dtype=torch.float32)
    logger.debug(f"Random test")
    x_test = X_test[torch.randint(0, X_test.size(0), (1,))] 
    model.eval()
    with torch.no_grad():
        y_pred = model(x_test)

    logger.info(f"Make the prediction for the test data")
    with torch.no_grad():
        y_pred = model(X_test)

    if args['target']:
        logger.info(f"Inverse the power transformer to the prediction")
        y_pred = y_pred.reshape(-1, 1)
        X_test = torch.concat((X_test, y_pred), axis=1)
        X_test = pt.inverse_transform(X_test)
        y_pred = X_test[:, -1]
    else: # exclude the target column
        y_pred = y_pred.numpy()

    logger.info(f"Make top 10% of the prediction to 1")
    y_pred_bin = (y_pred > np.percentile(y_pred, 90)).astype(np.int32)

    logger.info(f"Review the submission result")
    submission = pd.DataFrame({'ID': test_id, 'y': y_pred_bin, 'y_pred': y_pred})
    print(submission.sort_values(by='y_pred', ascending=False))

    logger.info(f"Save the submission result to submission.csv")
    submission.pop('y_pred')
    submission.to_csv(os.path.join(ROOT_DIR, 'dataset', 'submission.csv'), index=False)


