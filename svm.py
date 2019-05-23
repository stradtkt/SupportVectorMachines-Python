# -*- coding: utf-8 -*-
"""
Created on Thu May 23 17:56:14 2019

@author: Kevin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
cancer.keys()

df_feat = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])

from sklearn.model_selection import train_test_split
X = df_feat
y = cancer['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

from sklearn.svm import SVC
model = SVC()
model.fit(X_train, y_train)

pred = model.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))

from sklearn.grid_search import GridSearchCV
param_grid = {}