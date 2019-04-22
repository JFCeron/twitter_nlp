# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 17:25:05 2019

@author: Bananin
"""

import pandas as pd
import numpy as np
import pickle
from consolidate_split import consolidate_split
from preprocessing import preprocess
from naive_bayes import train_svm, train_svm_withC, predict
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
# split xlsx files root (split to facilitate classification)
split_root = "tweets/split/"
all_tweets_path = "tweets/tweets_venezuela.csv"

def main():
    # split xlsx files root (split to facilitate classification)
    split_root = "tweets/split/"
    all_tweets_path = "tweets/tweets_venezuela.csv"
    # consolidate split files
    classified, unclassified = consolidate_split(split_root)
    
    # MODEL TRAINING AND SELECTION
    
    print("Preprocessing classified tweets")
    train = preprocess(classified.Tuit)
    train["Clase"] = classified.Clase
    
    # keep the highest F1 score obtained for each preprocessing method
    scores = []
    for method in train.columns.values[range(len(train.columns)-1)]:
        print("Finding best model for preprocessing method: "+method)
        best_svm_method, score = train_svm(train[method], train["Clase"])
        if not [] or score > max(scores):
            best_svm = best_svm_method
            best_method = method
        scores.append(score)
    # store the best score for each method
    pd.DataFrame({"method":train.columns.values[range(len(train.columns)-1)],
                "score":scores}).to_csv("best_score_per_method.csv",index=False)
    # store the best model
    filehandler = open("best_svm.obj", 'wb') 
    pickle.dump(best_svm, filehandler)
    filehandler.close()
    
    # CLASSIFY ALL TWEETS
    
    # assign the human classification to tweets that were in the training set
    all_tweets = pd.read_csv(all_tweets_path)
    all_tweets = pd.merge(all_tweets, classified, left_on="CONTENT", right_on="Tuit", how="left")
    del all_tweets["Tuit"]
    
    # classify the rest
    unclassified_indices = np.isnan(all_tweets.Clase)
    # preprocess and classify
    # unique_unclassified = set(all_tweets.loc[unclassified_indices, "CONTENT"])
    # preprocessed_unclassified = preprocess(unique_unclassified)[["Tuit",best_method]]
    print("Preprocessing new tweets for classification")
    unclassified_preprocessed = preprocess(all_tweets.loc[unclassified_indices,"CONTENT"])
    print("Classifying")
    all_tweets.loc[unclassified_indices,"Clase"] = predict(best_svm, unclassified_preprocessed[best_method])

    # prepare data to by read by web app
    all_tweets.rename(columns={"Clase":"CATEGORY_NUMBER"}, inplace=True)
    category_names = pd.DataFrame(
            {"CATEGORY_NUMBER":range(1,7),"CATEGORY_LABEL":["Informativo","Solidario",
             "Protectivo","Critico de gobiernos","Prejuicioso","Argumentativo"]})
    all_tweets = pd.merge(all_tweets, category_names, left_on="CATEGORY_NUMBER", right_on="CATEGORY_NUMBER")
    # write data for web app
    all_tweets.to_csv("tweets/tweets_venezuela_clasificados.csv", index=False)
    
    # CONFUSION MATRIX ON A TEST SET
    
    # train an SVM on a subset of the classified data
    train_indices = np.random.choice(len(train), int(len(train)*0.8))
    svm = train_svm_withC(train.loc[train_indices, best_method], y=train.loc[train_indices, "Clase"], C=best_svm.C)
    y_hat = predict(svm, train.loc[[i for i in range(len(train)) if not i in train_indices], best_method])
    y = train.loc[[i for i in range(len(train)) if not i in train_indices], "Clase"].reset_index(drop=True)
    # draw a confusion matrix
    plt.matshow(confusion_matrix(y, y_hat))
    plt.colorbar()
    ax = plt.gca()
    ax.set_xticklabels(range(7))
    ax.set_yticklabels(range(7))
    plt.xlabel("Prediccion")
    plt.ylabel("Verdad")
    plt.savefig("confusion_coyuntura.png")
    # save report
    pd.DataFrame(classification_report(y, y_hat, output_dict = True)).to_csv("report_coyuntura.csv")
    
if __name__ == "__main__":
    main()