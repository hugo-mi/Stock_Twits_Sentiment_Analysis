#### COMMON ####
import pandas as pd
import numpy as np

import json
import requests
from datetime import datetime

### Data Viz ###
import seaborn as sns
import matplotlib.pyplot as plt

#### MONGO DB ####
from pymongo.mongo_client import MongoClient

### EVALUATION METRIC ###
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score

##################################################### 
# GLOBAL VARIABLES 
#####################################################
URI = "mongodb+srv://humichel:mHmAKDDHyJxc491o@cluster0.ywlmnoy.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
DB_NAME = "tweets_db"


##################################################### 
# MONGO DB
#####################################################

def test_connection_db(URI):
    # Create a new client and connect to the server
    client = MongoClient(URI)
    
    # Send a ping to confirm a successful connection
    try:
        client.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")
    except Exception as e:
        print(e)
        
        
def insert_to_mongodb(row, uri, db_name, collection_name):
    """
    Insert data into the collection
    """
    client = MongoClient(uri)
    db_tweets = client[db_name]
    collection = db_tweets[collection_name]

    # Insert row into MongoDB
    collection.insert_one(row.to_dict())
    
def clean_mongodb(uri, db_name, collection_name):
    """
    Clean the collection before inserting new data
    """
    client = MongoClient(uri)
    db_tweets = client[db_name]
    collection = db_tweets[collection_name]
    # Delete all existing documents from the collection before insert new data
    collection.delete_many({})
    print(db_name+"."+collection_name + " COLLECTION IS SUCCESSFULLY CLEANED \n\n")
    
# Iterate through each row and insert into MongoDB
def insert_tweets(df, uri, db_name, collection_name):
    """
    Insert tweet into the collection
    """
    # Clean the collection
    print("STARTING TO CLEAN DATA...")
    clean_mongodb(uri=URI, db_name=DB_NAME, collection_name="META")
    counter = 0
    print("STARTING TO INSERT DATA...")
    for index, row in df.iterrows():
        insert_to_mongodb(row, uri=URI, db_name=DB_NAME, collection_name="META")
        print("row: "+ str(index) + " inserted")
        counter = counter + 1
    print("\n\n " + str(counter) + " ROWS INSERTED SUCCESSFULLY INTO "+ db_name+"."+collection_name)
    

def get_tweets_from_db(uri, db_name, collection_name):
    # Function to connect to MongoDB and retrieve data
    client = MongoClient(uri)
    db_tweets = client[db_name]
    collection = db_tweets[collection_name]
    # Retrieve all documents from the collection
    cursor = collection.find({})
    # Convert documents to list of dictionaries
    data = list(cursor)
    return data

##################################################### 
# DATA COLLECTION (StockTwits) 
#####################################################
def collect_tweets(ticker="META", nb_url=10):

    headers = {'User-Agent': 'Mozilla/5.0 Chrome/39.0.2171.95 Safari/537.36'}

    rows = []
    print("STARTING TO COLLECT TWEET "+ ticker.upper() + "...\n\n")
    for i in range(0, nb_url):
        if i == 0:
            url = "https://api.stocktwits.com/api/2/streams/symbol/" + ticker + ".json"
        else:
            url = "https://api.stocktwits.com/api/2/streams/symbol/" + ticker + ".json?max=" + str(maxid)

        print("COLLECTNG FROM:... \n" + str(i+1) + ": " + url)

        r = requests.get(url, headers=headers)
        data = json.loads(r.content)

        maxid = data["cursor"]["max"]

        for m in data["messages"]:
            date = m["created_at"]
            content = m["body"]
            sentiment = ""
            if "Bearish" in str(m["entities"]["sentiment"]):
                sentiment = "bearish"
            if "Bullish" in str(m["entities"]["sentiment"]):
                sentiment = "bullish"
            if str(m["entities"]["sentiment"]) == "None":
                sentiment = "None"
            rows.append(( date, content, sentiment ))
            
    df_meta_tweets = pd.DataFrame(rows, columns=["date","content","true_sentiment"])
    
    # Filter out to keep all the tweets for which the sentiment is `bullish` or `bearish
    df_meta_tweets = df_meta_tweets[df_meta_tweets['true_sentiment'].isin(['bearish', 'bullish'])]
    
    print("\n" + str(len(df_meta_tweets)) + " TWEETS ARE SUCCESFULLY COLLECTED FROM THE TICKER" + ticker.upper())
    
    return df_meta_tweets


##################################################### 
# MODEL EVALUATION
#####################################################
def eval_model(true_labels, predicted_labels):

    # Compute classification report
    class_report = classification_report(true_labels, predicted_labels)

    print("######## Classification Report ########\n")
    print(class_report)

    accuracy = accuracy_score(true_labels, predicted_labels)
    print("######## Accuracy Score ########")
    print(round(accuracy,2))
    print()

    # Compute confusion matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    # Create a heatmap of the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['bullish', 'bearish'], yticklabels=['bullish', 'bearish'])
    plt.xlabel('Predicted Sentiment')
    plt.ylabel('True Sentiment')
    plt.title('Confusion Matrix')
    plt.show()