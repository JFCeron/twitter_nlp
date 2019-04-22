# -*- coding: utf-8 -*-
"""
Naive Bayes classification of tweets' classes using TFIDF features

@author: Bananin
"""

import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold, cross_val_score

# vectorizing object
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
# cross-validation object
k_fold = KFold(n_splits=10)
# different C to try for in cross-validation
all_C = [2**i for i in range(-6,6)]

# trains a SVM classifier 
def train_svm (content, y):
    # extract TF-IDF feature matrix    
    features = tfidf_vectorizer.fit_transform(content.astype('U'))
    
    # select C via cross-validation, according to mean F1 score
    best_score = 0
    best_svc = None
    for C in all_C:
        # support vector classifier
        svc = svm.SVC(C=C, kernel='linear')#, class_weight="balanced")
        score = np.mean(cross_val_score(svc, features, y, cv=k_fold, n_jobs=-1, scoring="f1_macro"))
        if score > best_score:
            best_score = score
            best_svc = svc
    best_svc.fit(features, y)
    return best_svc, best_score

def train_svm_withC (content, C, y):
    # extract TF-IDF feature matrix    
    features = tfidf_vectorizer.fit_transform(content.astype('U'))
    # support vector classifier
    svc = svm.SVC(C=C, kernel='linear')#, class_weight="balanced")
    svc.fit(features, y)
    return svc
        
# vectorizes and classifies using the given svc
def predict(svc, content):
    # extract TF-IDF feature matrix    
    features = tfidf_vectorizer.fit_transform(content.astype('U'))
    return svc.predict(features)