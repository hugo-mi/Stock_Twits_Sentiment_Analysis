#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 18:02:43 2024

@author: Utilisateur
"""
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
import regex as re 


df = pd.read_parquet("/Users/Utilisateur/Downloads/Stock_Twits_Sentiment_Analysis-main/balanced_250k.parquet")
df['true_sentiment'] = df['true_sentiment'].replace(['bullish', 'bearish'], [1, 0])
df = df.reset_index(drop=True)


df_balanced_250 = pd.concat([df.loc[df['true_sentiment'] == 1].sample(125000), df.loc[df['true_sentiment'] == 0].sample(125000)]).sample(250000)

# MNB model
model = MultinomialNB(alpha=0.1, fit_prior=True, class_prior=None)

def replace_slang(text, slang_dict):
    pattern = re.compile(r'\b(' + '|'.join(re.escape(key) for key in slang_dict.keys()) + r')\b')
    return pattern.sub(lambda x: slang_dict[x.group()], text)

def load_slang_table(filepath):
    slang_dict = {}
    with open(filepath, 'r', encoding='ISO-8859-1') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                slang, meaning = parts
                slang_dict[slang] = meaning
    return slang_dict


slang_dict = load_slang_table("/Users/Utilisateur/Downloads/Stock_Twits_Sentiment_Analysis-main/SlangLookupTable.txt")
df_balanced_250['content_cleaned_slang'] = df_balanced_250['content_cleaned'].apply(lambda x: replace_slang(x, slang_dict))
vectorizer = CountVectorizer(stop_words='english')
X_original = vectorizer.fit_transform(df_balanced_250['content_cleaned'])
X_slang = vectorizer.fit_transform(df_balanced_250['content_cleaned_slang'])
y = df_balanced_250['true_sentiment']

def evaluate_model(X, y):
    accuracy = cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()
    mcc = cross_val_score(model, X, y, cv=5, scoring='matthews_corrcoef').mean()
    return accuracy, mcc

accuracy_original, mcc_original = evaluate_model(X_original, y)
accuracy_slang, mcc_slang = evaluate_model(X_slang, y)

print("Original Text - Accuracy:", accuracy_original, "MCC:", mcc_original)
print("Slang-Replaced Text - Accuracy:", accuracy_slang, "MCC:", mcc_slang)
