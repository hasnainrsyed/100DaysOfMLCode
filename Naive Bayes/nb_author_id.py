#!/usr/bin/python3

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


##############################################################
# Enter Your Code Here

nb_clf = GaussianNB()
# Training the classifier with the training data
t0 = time()
nb_clf.fit(features_train, labels_train)
print("Training Time:", round(time()-t0, 3), "s")

t0 = time()
pred = nb_clf.predict(features_test)
print("Predicting Time:", round(time()-t0, 3), "s")

X = features_train
Y = labels_train
accuracy = nb_clf.score(X,Y)
print(accuracy)

acc = accuracy_score(pred, labels_test)
print(acc)

##############################################################