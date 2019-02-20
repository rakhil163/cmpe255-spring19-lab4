#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 21:11:46 2019

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
        

   
#using kNN Classifier        
knn=KNeighborsClassifier()
knn.fit(train,labels)
y=knn.predict(test)
print (accuracy_score(y,templist2))