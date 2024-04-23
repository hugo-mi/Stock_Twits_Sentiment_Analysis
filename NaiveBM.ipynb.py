#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 15:25:06 2024

@author: Utilisateur
"""

#### NLP LIBS ###
import nltk
import numpy as np
import pandas as pd

### Import utils function ###
from utils import get_tweets_from_db, eval_model, URI, DB_NAME
from cleaning_utils import full_preprocess

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, matthews_corrcoef

# Retrieve data from MongoDB
data_from_mongodb = get_tweets_from_db(uri=URI, db_name=DB_NAME, collection_name="AAPL")

# Create DataFrame from retrieved data
df = pd.DataFrame(data_from_mongodb)

df["cleaned_content"] = df["content"].apply(full_preprocess)
df

#### ici les none sont encore de la partie donc applied le code post le bon pre process ET DONC COMPARE LES DATASETS 

# Vectorization
vectorizer = CountVectorizer(stop_words='english') 
X = vectorizer.fit_transform(df['cleaned_content'])
y = df['true_sentiment']

### NBM is MultinomialNB
model = MultinomialNB(alpha=0.1, fit_prior=True, class_prior=None)

def evaluate_model_on_different_sizes(X, y, model, sizes, cv=5):
    results = []
    for size in sizes:
        X_subset, y_subset = X[:size], y[:size]
        accuracy = cross_val_score(model, X_subset, y_subset, cv=cv, scoring='accuracy').mean()
        mcc = cross_val_score(model, X_subset, y_subset, cv=cv, scoring='matthews_corrcoef').mean()
        results.append({
            'size': size,
            'accuracy': accuracy,
            'mcc': mcc
        })
    return pd.DataFrame(results)

sizes = [25, 50, 100]  #c'est la hess des datassss, a changer par les bonnes : 500/1000/2500/5000/10000/25000/50000/100000/250000/500000/1000000
results = evaluate_model_on_different_sizes(X, y, model, sizes)

