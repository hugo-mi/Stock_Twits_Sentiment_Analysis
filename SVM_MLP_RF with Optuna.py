
import pandas as pd
import numpy as np
import nltk
import optuna
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier


df_apple_tweets = pd.read_parquet('balanced_250k.parquet')
df_apple_tweets['cleaned_content'] = df_apple_tweets['content_cleaned']

X = df_apple_tweets["cleaned_content"]
y = df_apple_tweets['true_sentiment']

vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=25, tokenizer=nltk.TweetTokenizer().tokenize)
X_vectorized = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Optuna
def objective_svm(trial):
    C = trial.suggest_loguniform('C', 1e-4, 1e4)
    classifier = svm.LinearSVC(C=C, random_state=42)
    classifier.fit(X_train, y_train)
    return accuracy_score(y_test, classifier.predict(X_test))

def objective_mlp(trial):
    alpha = trial.suggest_loguniform('alpha', 1e-5, 1e-1)
    learning_rate_init = trial.suggest_loguniform('learning_rate_init', 1e-5, 1e-1)
    classifier = MLPClassifier(random_state=42, max_iter=50, tol=0.001, 
                               early_stopping=True, n_iter_no_change=5,
                               alpha=alpha, learning_rate_init=learning_rate_init)
    classifier.fit(X_train, y_train)
    return accuracy_score(y_test, classifier.predict(X_test))

def objective_rf(trial):
    n_estimators = trial.suggest_int('n_estimators', 10, 200)
    max_depth = trial.suggest_int('max_depth', 5, 50)
    classifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    classifier.fit(X_train, y_train)
    return accuracy_score(y_test, classifier.predict(X_test))

# Optuna studies for each model
study_svm = optuna.create_study(direction='maximize')
study_svm.optimize(objective_svm, n_trials=10)

study_mlp = optuna.create_study(direction='maximize')
study_mlp.optimize(objective_mlp, n_trials=10)

study_rf = optuna.create_study(direction='maximize')
study_rf.optimize(objective_rf, n_trials=10)

print("Best SVM Model:", study_svm.best_params)
print("Best SVM Accuracy:", study_svm.best_value)

print("Best MLP Model:", study_mlp.best_params)
print("Best MLP Accuracy:", study_mlp.best_value)

print("Best RF Model:", study_rf.best_params)
print("Best RF Accuracy:", study_rf.best_value)
