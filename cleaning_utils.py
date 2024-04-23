#%%
import pandas as pd
import numpy as np
import regex as re
from nltk.stem import PorterStemmer
from nltk.tag import StanfordPOSTagger
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import CountVectorizer

test = 'hello, i think $META will go up and as @bolosse_du_75 said earlier on https.alpha_trader.com'

def replace_tags(text ,ticker=True, urls=True, users=True):
    
    if str(text) == 'nan':
        return ''

    if ticker:
        text = re.sub('\$[A-Z]{1,6}', 'cashtag', text)

    if urls:
        text = re.sub('http[^ ]*', 'linktag', text)

    if users:
        text = re.sub('@[^ ]*', 'usertag', text)

    return text 

def tokenize_text(text):
    tokenizer = TweetTokenizer()

    return tokenizer.tokenize(text)

def stem_text(tokens):
    stemmer = PorterStemmer()

    return [stemmer.stem(w) for w in tokens]

def remove_stop_words(tokens, stop_words=None):

    if str(stop_words) == 'None':
        stop_words = stopwords.words('english')

    return [w for w in tokens if w not in stop_words]

def clean_unnecessary_punctuation(tokens): 

    signs_to_keep ='! ? % + - = : ; ) ( ]'.split()

    return [w for w in tokens if (w.isalnum()) | (w in signs_to_keep)]

def full_preprocess(text, remove_stops=True):

    text = replace_tags(text)
    text = clean_unnecessary_punctuation(stem_text(tokenize_text(text)))

    if remove_stops:
        text = remove_stop_words(text)

    return text


def get_grams(tokens, ngrams=(1,1), min_frequency=0):

    vectorizer = CountVectorizer(input='content', ngram_range=ngrams, min_df=min_frequency)
    
    return vectorizer.fit_transform(tokens) # can add .toarray() for numpy representation

