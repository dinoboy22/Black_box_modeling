#!/usr/bin/env python
# coding: utf-8

import os
import sys
sys.path.append('./')
sys.path.append('../')

import pandas as pd
import optuna
import config

# import numpy as np
# import matplotlib.pyplot as plt
# from libs.plots import plot_heatmap, plot_feature, plot_feature_distribution, plot_feature_to_target

# from sklearn.base import TransformerMixin, BaseEstimator
# from sklearn.pipeline import Pipeline, FeatureUnion
# from sklearn.linear_model import LinearRegression
# from xgboost import XGBRegressor
# from catboost import CatBoostRegressor
# import lightgbm as lgbm

# from optuna import Trial, visualization

# from pytorch_tabnet.tab_model import TabNetRegressor
# from sklearn.model_selection import KFold
# import torch

ROOT_DIR = '.' if os.path.exists('config') else '..' 
train_csv = os.path.join(ROOT_DIR, 'dataset', 'train.csv')
test_csv = os.path.join(ROOT_DIR, 'dataset', 'test.csv')

train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)

# train_df = train_df.drop(columns=['ID'])
# test_df = test_df.drop(columns=['ID'])

for col in train_df.columns :
    cnt = train_df[col].value_counts()
    print(f"{cnt}\n")

# train_df.isnull().sum()


# list 는 keyword 이므로 변수명으로 사용하지 않는 것이 좋다.
# list = []
# for i in range (100):
#     list.append(i+1)
labels = list(range(1,101)) # [1, 2, 3, ..., 100]

for col in train_df.columns :
    if col != 'ID' and col != 'y':
        train_df[col] = pd.qcut(train_df[col], q=100, labels=labels)

for col in test_df.columns :
    if col != 'ID' and col != 'y':
        test_df[col] = pd.qcut(test_df[col], q=100, labels=labels)


quantile_value = train_df['y'].quantile(0.9)

train_df['y'] = train_df['y'].apply(lambda x: 1 if x >= quantile_value else 0)


from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from catboost import CatBoostClassifier

# RANDOM_STATE = 110
# THRESHOLD = 0.5
RANDOM_STATE = config.CAT_BOOST_RANDOM_STATE
THRESHOLD = config.CAT_BOOST_THRESHOLD

def objectiveCatBoost(trial, x_tr, y_tr, x_val, y_val):
    param = {
        'iterations': trial.suggest_int('iterations', 800, 5000),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.2),
        'depth': trial.suggest_int('depth', 4, 13),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 5),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10, log=True),
        'random_strength': trial.suggest_float('random_strength', 0, 10),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.1, 10),
        'border_count': trial.suggest_int('border_count', 128, 300),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 10),
        'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),
        'random_seed': RANDOM_STATE,
        'eval_metric': 'F1',
        'logging_level': 'Silent',
        'boosting_type': 'Plain'
    }
    
    cat_features = []
    for i in x_tr.columns:
        cat_features.append(i)
    
    model = CatBoostClassifier(**param, cat_features=cat_features)
    model.fit(x_tr, y_tr)
    pred_proba = model.predict_proba(x_val)[:, 1]  # 양성 클래스 확률
    pred = (pred_proba >= THRESHOLD).astype(int)  # 스레드홀드에 따른 예측
    
    score = f1_score(y_val, pred, average="binary")
    return score

# 데이터셋 분할
x_train, x_val, y_train, y_val = train_test_split(
    train_df.drop(["y","ID"], axis=1),  
    train_df["y"],
    test_size=0.2,
    shuffle=True,
    random_state=RANDOM_STATE,
)

# 하이퍼 파라미터 튜닝
study = optuna.create_study(
    direction='maximize', 
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
)

study.optimize(
    lambda trial: objectiveCatBoost(trial, x_train, y_train, x_val, y_val), 
    n_trials=100
)

print('Best trial: score {}, \nparams {}'.format(study.best_trial.value, study.best_trial.params))


def fit_all_train_data_function(model_name, data, target_column='y', ID='ID', **params):
    # target_column이 data에 존재하는지 확인
    if target_column not in data.columns:
        raise ValueError(f"'{target_column}' 컬럼이 데이터프레임에 존재하지 않습니다.")
    
    # 범주형 피처 자동 선택 (데이터 타입이 'category' 또는 'object'인 경우)
    cat_features = [col for col in data.columns if data[col].dtype.name in ['category']]

    print(cat_features)
    
    # target_column을 제외한 학습 데이터 생성
    train_df = data.drop(columns=[target_column, ID], axis=1)
    
    # 모델 선택 및 하이퍼파라미터 설정
    model = CatBoostClassifier(**params, cat_features=cat_features)
    
    # 모델 학습
    model.fit(train_df, data[target_column])
    
    # 학습 완료 메시지 출력
    print(f'{model_name} 모델이 주어진 데이터로 학습 완료')
    
    return model  # 학습된 모델 반환

param = {
    'iterations' : 4375, 
    'learning_rate' : 0.010420215040093828, 
    'depth' : 10, 
    'min_data_in_leaf' : 3,
    'l2_leaf_reg' : 1.1241408183988555,
    'random_strength' : 7.071934739361616, 
    'bagging_temperature' : 9.151169182653845, 
    'border_count' : 128, 
    'scale_pos_weight' : 3.103993405620786,
    'grow_policy' : 'SymmetricTree',

    'random_state' : RANDOM_STATE,
    'eval_metric' : 'F1',
    'logging_level' : 'Silent',
    'boosting_type' : 'Plain'
}


cat_features = [col for col in train_df.columns if train_df[col].dtype.name in ['category']]

train_df = train_df.drop(columns=['ID', 'y'], axis=1)


model = CatBoostClassifier(**param, cat_features=cat_features)
model.fit(train_df, train_df['y'])

df_test = test_df.drop(columns=['ID'],axis=1)
preds = model.predict_proba(df_test)

pd_preds = pd.DataFrame(preds)

sum = 0
for i in pd_preds[1]:
    if i >= 0.5:
        sum = sum+1

pd_preds[1] = pd_preds[1]*100

sample_submission_csv_path = os.path.join(
    ROOT_DIR, 
    'dataset', 
    'sample_submission.csv'
)

submission_csv_path = os.path.join(
    ROOT_DIR, 
    'dataset', 
    'submission.csv'
)

submission = pd.read_csv(sample_submission_csv_path)
submission['y'] = pd_preds[1]
submission.to_csv(submission_csv_path, index=False)

