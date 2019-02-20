#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 21:08:39 2019

@author: rakhil163
"""

from sklearn.linear_model import SGDClassifier
import numpy as np
from sklearn.metrics import accuracy_score
from mnist import MNIST
from sklearn.neighbors import KNeighborsClassifier
mndata = MNIST('img')

train,labels=mndata.load_training()
test,labels_test=mndata.load_testing()
templist=[]
for i in labels.tolist():
    if(i%2==0):
        templist.append("Even")
    else:
        templist.append("Odd")
    
templist2=[]
    
for i in labels_test.tolist():
    if(i%2==0):
        templist2.append("Even")
    else:
        templist2.append("Odd")
        
        

#using BinaryClassifier
sgd_clf = SGDClassifier(max_iter=5, tol=-np.infty, random_state=42)
sgd_clf.fit(train, labels)
x=sgd_clf.predict(test)
print (accuracy_score(x,templist2))

