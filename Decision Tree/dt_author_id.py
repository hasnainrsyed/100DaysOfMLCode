#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
### your code goes here ###
### The number of features determine the complexity of the decision tree
X = features_train
Y = labels_train
Z = features_test
V = labels_test
print ("No. of features:", len(X[0])) #The number of features considered

clf = DecisionTreeClassifier(min_samples_split=40)
t0 = time()
clf.fit(X,Y)
print("Training Time:", round(time()-t0, 3), "s")
t0 = time()
pred = clf.predict(Z)
print("Predicting Time:", round(time()-t0, 3), "s")
accuracy = accuracy_score(pred,V)
print(accuracy)
#########################################################
