#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: guowei
"""

#import all packages used in this program
import numpy as np
import pandas as pd
from sklearn import datasets 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap 
import matplotlib.pyplot as plt
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score

#get iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

#part1: Random test train splits
in_sample_scores = []
out_of_sample_scores = []
random_state = np.arange(1,11,1).tolist()

for n in range(1,11):
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size = 0.1, 
                                                        random_state = n)
    
    tree = DecisionTreeClassifier(criterion='gini',max_depth=4, random_state=1)
    tree.fit(X_train, y_train)
    y_pred_train = tree.predict(X_train)
    y_pred_test = tree.predict(X_test)
    
    in_sample_scores.append(accuracy_score(y_train,y_pred_train))
    out_of_sample_scores.append(accuracy_score(y_test,y_pred_test))
    
 
#report the scores, mean and std in a dataframe 
scores = pd.DataFrame({'random_state':random_state,
                       'in_sample_scores':in_sample_scores,
                       'out_of_sample_scores':out_of_sample_scores})
scores = scores.round({'random_state': 0, 'in_sample_scores': 3, 'out_of_sample_scores': 3})
scores_desc = scores.describe()
scores_stat = pd.concat([scores,scores_desc])
scores_stat = scores_stat.drop(['count', 'min','25%', '50%','75%', 'max'])
scores_stat.loc['mean','random_state'] = '--'
scores_stat.loc['std','random_state'] = '--'
print(scores_stat)

del scores, scores_desc, in_sample_scores, out_of_sample_scores,random_state


#part2 Cross validation
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.1, 
                                                    random_state = 11)
cv_socres = cross_val_score(tree, X_train, y_train, cv=10)
print('Cross Validation scores:',cv_socres.tolist())
print()
print('Mean of CV scores:',cv_socres.mean())
print()
print('Std of CV scores:',cv_socres.std())
print()

y_pred = tree.predict(X_test)
scores = accuracy_score(y_test, y_pred) 
print('scores: ',out_sample_accuracy)
print()

###
print("My name is Wei Guo")
print("My NetID is: weiguo6")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")

 