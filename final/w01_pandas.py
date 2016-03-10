# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


data= pd.read_csv('features.csv', index_col='match_id')
data_test = pd.read_csv('features_test.csv', index_col='match_id')

vals = list(set(data.columns.values) - set(data_test.columns.values))
X_full = data[list(data_test.columns.values)].dropna(axis=1, how='any')
missing_cols = list(set(data.columns.values) - set(X_full.columns.values))
y = data['radiant_win']