# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 09:44:49 2019

@author: Bananin
"""

import pandas as pd
from unidecode import unidecode
from spellchecker import SpellChecker
import nltk
import re

def preprocess (content):

    # spanish stemmer
    stemmer = nltk.stem.SnowballStemmer("spanish")
    # spanish stopwords
    stopwords = nltk.corpus.stopwords.words("spanish")
    # spanish spell-checking tool
    spell = SpellChecker(language="es", distance=1)
    
    # we'll save differently preprocessed versions of the data to compare
    letters_only = list()
    no_stopwords = list()
    spelling_corrected = list()
    stemmed = list()
    
    for tweet in content:
        # to lower case
        tweet = str(tweet).lower()
        # no accents
        tweet = unidecode(tweet)
        # no urls
        tweet = re.sub("(http|www)[^ ]*","",tweet)
        # letters only
        tweet = re.sub("[^a-zA-Z]", " ", tweet)
        letters_only.append(tweet)
        # word tokenization
        words = nltk.word_tokenize(tweet)
        # remove spanish stopwords
        words = [w for w in words if w not in stopwords]
        no_stopwords.append(" ".join(words))
        # correct spelling at a distance of one character
        words = [spell.correction(w) for w in words]
        spelling_corrected.append(" ".join(words))
        # spanish stemming
        words = [stemmer.stem(w) for w in words]
        stemmed.append(" ".join(words))
    
    # return the differently preprocessed contents
    return pd.DataFrame(data = {"letters_only":letters_only, "no_stopwords":no_stopwords,
                                "spelling_corrected":spelling_corrected, "stemmed":stemmed})
    