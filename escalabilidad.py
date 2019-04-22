# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 23:20:02 2019

@author: Bananin
"""

import time
import pandas as pd
import numpy as np
from preprocessing import preprocess
from naive_bayes import train_svm, predict

# volumen de cada dataset en las pruebas de escalabilidad
N = [5000, 20000, 50000, 100000, 1000000]
# nuestro dataset no es tan grande como es necesario originalmente
tweets = pd.read_csv("tweets_venezuela_clasificados.csv")
tweets = tweets.append(tweets)
# dataframes de resultados
tiempos = pd.DataFrame({"preprocesamiento":[0]*len(N),"entrenamiento":[0]*len(N),
                        "clasificacion":[0]*len(N)}, index=N)
best_method = "stemmed"
for n in N:
    print("Prueba de tamano n="+str(n))
    dataset = tweets.iloc[range(n)]

    print("Preprocesando")
    start_pre = time.time()
    preprocessed = preprocess(dataset.CONTENT)
    preprocessed["Clase"] = np.random.choice(range(1,7), len(preprocessed))
    tiempos.loc[n, "preprocesamiento"] = time.time() - start_pre
    tiempos.to_csv("escalabilidad/tiempos.csv")

    print("Entrenando")
    start_train = time.time()
    best_svm, score = train_svm(preprocessed[best_method], preprocessed["Clase"])
    tiempos.loc[n, "entrenamiento"] = time.time() - start_train
    tiempos.to_csv("escalabilidad/tiempos.csv")

    print("Clasificando")
    start_class = time.time()
    predict(best_svm, preprocessed[best_method])
    tiempos.loc[n, "clasificacion"] = time.time() - start_class
    tiempos.to_csv("escalabilidad/tiempos.csv")
