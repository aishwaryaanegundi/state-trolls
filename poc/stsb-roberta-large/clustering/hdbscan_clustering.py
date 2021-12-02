import numpy as np 
import pandas as pd
import csv
import os
import re
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import random
import emoji
import string
import json
import nltk

from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine

# list of all the dataset files
dataset_paths = ["../../../datasets/russia_052020_tweets_csv_hashed_2.csv", 
         "../../../datasets/russian_linked_tweets_csv_hashed.csv", 
         "../../../datasets/ira_tweets_csv_hashed.csv", 
         "../../../datasets/russia_201906_1_tweets_csv_hashed.csv"]

# path to store the entire combined dataset
combined_dataset_path = "../datasets/russian_trolls.csv"

# returns a pandas dataframe consisting of entries from all the dataset files
def get_combined_dataset(paths):
    data = pd.concat((pd.read_csv(file) for file in tqdm(paths)))
    return data

data = get_combined_dataset(dataset_paths)
print("Number of tweets in the dataset: ", data.shape[0])

# # extracts just the english tweets by using the language tag
# is_english_tweet = data['tweet_language'] == 'en'
# english_data = data[is_english_tweet]

# print("Number of English tweets in the dataset: ", english_data.shape[0])
# english_tweet_data = english_data[['tweetid', 'tweet_text']]

# takes list of tweets as input and returns list of pre-processed tweets as output
def preprocess(tweets):
    processed_tweets = []
    for tweet in tweets:
        result = re.sub(r"http\S+", "", tweet)
        result = re.sub(r"RT @\S+", "", result)
        result = re.sub(r"@\S+", "", result)
        result = re.sub(emoji.get_emoji_regexp(), "", result)
        result_removed_punctuation = result.translate(str.maketrans('', '', string.punctuation))
        result = re.sub(r"[^a-zA-Z0-9 ]", "", result_removed_punctuation)
        processed_tweets.append(result)
    return processed_tweets

# Filtering data between 01.01.2019 to 01.07.2017
data['tweet_time'] = pd.to_datetime(data['tweet_time'], format = '%Y-%m-%d')
start_date = '2016-01-01'
end_date = '2017-07-01'
mask = (data['tweet_time'] > start_date) & (data['tweet_time'] <= end_date)
data = data.loc[mask]
print("Number of tweets in the dataset after filtering: ", data.shape[0])


# extracts just the english tweets by using the language tag
is_english_tweet = data['tweet_language'] == 'en'
f_english_data = data[is_english_tweet]
print("Number of English tweets in the filtered dataset: ", f_english_data.shape[0])



# In[145]:


is_retweet = f_english_data['retweet_tweetid'].notnull()
f_english_data = f_english_data[~is_retweet]
print("Number of entries in the dataset after removing retweets: ", f_english_data.shape[0])


# In[148]:


f_english_tweet_data = f_english_data[['tweetid', 'tweet_text']]

tweets = f_english_tweet_data['tweet_text']
tweets = preprocess(tweets)

f_english_tweet_data = f_english_tweet_data.assign(processed_tweets = tweets)

# removes the entries having empty string after preprocessing
is_not_empty_string = f_english_tweet_data['processed_tweets'].apply(lambda x: not str.isspace(x))
f_english_tweet_data = f_english_tweet_data[is_not_empty_string]
print("Number of english tweets after filtering and preprocessing before dropping the duplicates: ", f_english_tweet_data.shape[0])
f_english_tweet_data.drop_duplicates(subset ="tweetid", keep = 'first', inplace = True)
f_english_tweet_data.drop_duplicates(subset ="processed_tweets", keep = 'first', inplace = True)
f_english_tweet_data = f_english_tweet_data.reset_index()
print("Number of english tweets after filtering and preprocessing and dropping the duplicates: ", f_english_tweet_data.shape[0])


# In[149]:



encodings = np.load('../encodings_stsb_roberta_large_fixed_retweet.npy')

# normalizes the vectors in ndarray row wise
def normalize_rows(x: np.ndarray):
    return x/np.linalg.norm(x, ord=2, axis=1, keepdims=True)


# In[150]:


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', -1)


# In[151]:

import numpy as np
             
X = normalize_rows(encodings)
start_time_c = time.perf_counter()

import hdbscan
clusterer = hdbscan.HDBSCAN(min_cluster_size=5, core_dist_n_jobs =50)
cluster_labels = clusterer.fit_predict(X)

end_time_c = time.perf_counter()
print("Time taken for clustering is :", (end_time_c - start_time_c)/60.0, " minutes.")



ref_df = pd.DataFrame()
ids = ((f_english_tweet_data)['tweetid']).astype(str).tolist()
tweets = (f_english_tweet_data)['processed_tweets'].tolist()
ref_df['tweet_text'] = tweets
ref_df['tweetid'] = ids
ref_df['cluster_label'] = cluster_labels



# In[156]:


ref_df = ref_df.sort_values('cluster_label')


# In[157]:


ref_df.to_csv('hdbscanEntireRemovedRetweets.csv')

