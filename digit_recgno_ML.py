# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 15:50:30 2018

@author: Shivam-PC
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')


x = data_train.iloc[:, 1:].values
y = data_train.iloc[:, 0].values

x_test = data_test.iloc[:, :].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


#  AdaBoostClassifier accuracy 0.738
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
clf = AdaBoostClassifier()
clf.fit(X_train, y_train) 
print(clf.score(X_test, y_test))

# RandomForestClassifier accuracy 0.941
clf = RandomForestClassifier()
clf.fit(X_train, y_train) 
print(clf.score(X_test, y_test))

# Decision tree accuracy 0.844 percent
from sklearn import tree
clf = tree.DecisionTreeClassifier(min_samples_split=40)
clf = clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

from sklearn import svm
clf = svm.SVC(kernel='rbf', C=0.0001)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
