#!/usr/bin/env python
# coding: utf-8

import os
import sys
sys.path.append('./')
sys.path.append('../')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from libs.plots import plot_heatmap, plot_feature, plot_feature_distribution, plot_feature_to_target

import config

ROOT_DIR = '.' if os.path.exists('config') else '..' 
train_csv = os.path.join(ROOT_DIR, 'dataset', 'train.csv')
train_df = pd.read_csv(train_csv)
train_df = train_df.drop(columns=['ID'])

# columns to inspect
# columns = train_df.columns[:-1]
columns = ['x_0', 'x_1', 'x_2', 'x_3', 'x_4']

for col in columns:
    plot_feature(train_df, col)
    plot_feature_distribution(train_df, col)
    plot_feature_to_target(train_df, col, 'y')

plot_heatmap(train_df, columns)
plt.show()




