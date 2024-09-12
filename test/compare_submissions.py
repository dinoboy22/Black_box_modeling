#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append('./')
sys.path.append('../')

import pandas as pd
import numpy as np
import re
import os
from simple_term_menu import TerminalMenu
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

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
    ap.add_argument('-v', '--verbose', type=int, default=0, help='verbose level')
    ap.add_argument('--debug', default=False, action=BooleanOptionalAction, help='debug message')

    args = vars(ap.parse_args())
    if args['verbose']:
        logger.setLevel(logging.DEBUG)

    logger.info("Started...")
    logger.debug(f"Argument: {args}")
    
    ROOT_DIR = '.' if os.path.exists('config') else '..' 
    submission_dir = os.path.join(ROOT_DIR, 'dataset')
    files = [file for file in [files for _,_,files in os.walk(submission_dir)]]
    submission_files = [file for file in files[0] if re.search('^submission(.)+csv$', file)]
    submission_files.sort()
    # print(*submission_files, sep='\n')

    logger.info("Select the first submission file: ")
    terminal_menu = TerminalMenu(submission_files)
    choice_index = terminal_menu.show()
    submission_1 = submission_files[choice_index]
    logger.info(f'Selected {submission_1}')

    logger.info("Select the second submission file: ")
    terminal_menu = TerminalMenu(submission_files)
    choice_index = terminal_menu.show()
    submission_2 = submission_files[choice_index]
    logger.info(f'Selected {submission_2}')

    logger.info(f"Compare {submission_1} and {submission_2}")

    submission_1 = pd.read_csv(os.path.join(submission_dir, submission_1))
    submission_2 = pd.read_csv(os.path.join(submission_dir, submission_2))

    submissions = np.stack([submission_1['y'], submission_2['y']], axis=1)
    submission_1 = submissions[:, 0]
    submission_2 = submissions[:, 1]

    cr = classification_report(submission_1, submission_2)
    cm = confusion_matrix(submission_1, submission_2)

    logger.info(f'\n{cr}\n')
    logger.info(f'\n{cm}\n')
    
