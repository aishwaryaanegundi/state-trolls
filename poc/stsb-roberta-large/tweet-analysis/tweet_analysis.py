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

dataset_paths = ["/INET/state-trolls/work/state-trolls/datasets/russia_052020_tweets_csv_hashed_2.csv", 
         "/INET/state-trolls/work/state-trolls/datasets/russian_linked_tweets_csv_hashed.csv", 
         "/INET/state-trolls/work/state-trolls/datasets/ira_tweets_csv_hashed.csv", 
         "/INET/state-trolls/work/state-trolls/datasets/russia_201906_1_tweets_csv_hashed.csv"]

# path to store the entire combined dataset
combined_dataset_path = "../datasets/russian_trolls.csv"

# returns a pandas dataframe consisting of entries from all the dataset files
def get_combined_dataset(paths):
    data = pd.concat((pd.read_csv(file) for file in tqdm(paths)))
    return data

data = get_combined_dataset(dataset_paths)
print("Number of tweets in the dataset: ", data.shape[0])

# extracts just the english tweets by using the language tag
is_english_tweet = data['tweet_language'] == 'en'
english_data = data[is_english_tweet]

print("Number of English tweets in the dataset: ", english_data.shape[0])
english_tweet_data = english_data[['tweetid', 'tweet_text']]

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
f_english_tweet_data = f_english_data[['tweetid', 'tweet_text']]

tweets = f_english_tweet_data['tweet_text']
tweets = preprocess(tweets)

f_english_tweet_data = f_english_tweet_data.assign(processed_tweets = tweets)

# removes the entries having empty string after preprocessing
is_not_empty_string = f_english_tweet_data['processed_tweets'].apply(lambda x: not str.isspace(x))
f_english_tweet_data = f_english_tweet_data[is_not_empty_string]
print("Number of english tweets after filtering and preprocessing before dropping the duplicates: ", f_english_tweet_data.shape[0])
f_english_tweet_data.drop_duplicates(subset ="tweetid", keep = 'first', inplace = True) 
f_english_tweet_data = f_english_tweet_data.reset_index()
print("Number of english tweets after filtering and preprocessing and dropping the duplicates: ", f_english_tweet_data.shape[0])
twitter_data = f_english_tweet_data

import numpy as np 
import pandas as pd
import string
import json
import glob
import os


# with open('/INET/state-trolls/work/state-trolls/reddit_dataset/comments/scores/RC_2016-01.bz2.decompressed/1_scores_stsb.txt', 'r') as content_file:
#     content = content_file.read()
#     json_data = content.replace('][',',')
#     j_object = json.loads(json_data)

# import collections
# j_df = pd.DataFrame(j_object)
# counts = collections.Counter(j_df['tweet_id'].tolist())
# counts_sorted = counts.most_common()
# c_df = pd.DataFrame(counts_sorted, columns=['tweet_id', 'frequency'])

counts10df = pd.read_csv('/INET/state-trolls/work/state-trolls/poc/stsb-roberta-large/tweet-analysis/counts-10.csv')
dicts = []

for index,row in counts10df.iterrows(): 
    text = (f_english_tweet_data.loc[f_english_tweet_data['tweetid'] == row['tweet_id']])['processed_tweets']
    dicts.append({'text': text, 'count': row['frequency']})
    if (index % 10000 == 0):
        dicts_df = pd.DataFrame(dicts)
        dicts_df.to_csv('m1i20.csv')
        
print(len(dicts))
dicts_df = pd.DataFrame(dicts)
dicts_df.to_csv('m1i20.csv')
