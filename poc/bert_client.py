import numpy as np 
import pandas as pd
import csv
import os
import re
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import pickle
import random
import string

from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine

# displays all columns and rows when asked to print
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', -1)

# list of all the dataset files
dataset_paths = ["../datasets/russia_052020_tweets_csv_hashed_2.csv", 
         "../datasets/russian_linked_tweets_csv_hashed.csv", 
         "../datasets/ira_tweets_csv_hashed.csv", 
         "../datasets/russia_201906_1_tweets_csv_hashed.csv"]

# path to store the entire combined dataset
combined_dataset_path = "../datasets/russian_trolls.csv"

# returns a pandas dataframe consisting of entries from all the dataset files
def get_combined_dataset(paths):
    data = pd.concat((pd.read_csv(file) for file in paths))
    return data

data = get_combined_dataset(dataset_paths)
print("Number of tweets in the combined dataset: ", data.shape[0])

# extracts just the english tweets by using the language tag
is_english_tweet = data['tweet_language'] == 'en'
english_data = data[is_english_tweet]
print("Number of English tweets in the dataset: ", english_data.shape[0])
english_tweet_data = english_data[['tweetid', 'tweet_text']]

def remove_url(tweet):
    result = re.sub(r"http\S+", "", tweet)
    return result

def remove_mentions(tweet):
    result = re.sub(r"@\S+", "", tweet)
    return result

def remove_retweet(tweet):
    result = re.sub(r"RT @\S+", "", tweet)
    return result

# takes list of tweets as input and returns list of pre-processed tweets as output
def preprocess(tweets):
    processed_tweets = []
    for tweet in tweets:
        result = remove_mentions(remove_retweet(remove_url(tweet)))
        processed_tweets.append(result)
    return processed_tweets

tweets = english_tweet_data['tweet_text']
tweets = preprocess(tweets)

english_tweet_data = english_tweet_data.assign(processed_tweets = tweets)

# removes the entries having empty string after preprocessing
is_not_empty_string = english_tweet_data['processed_tweets'].apply(lambda x: not str.isspace(x))
english_tweet_data = english_tweet_data[is_not_empty_string]
print("Number of english tweets after preprocessing and before resetting index: ", english_tweet_data.shape[0])
english_tweet_data = english_tweet_data.reset_index()
print("Number of english tweets after preprocessing: ", english_tweet_data.shape[0])

# Filtering data between 01.01.2019 to 01.07.2017
print("Number of tweets in the dataset: ", data.shape[0])
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
f_english_tweet_data = f_english_tweet_data.reset_index()
print("Number of english tweets after filtering and preprocessing: ", f_english_tweet_data.shape[0])
indexes = f_english_tweet_data['index'].tolist()
final_data = english_tweet_data[english_tweet_data['index'].isin(indexes)]
f_indexes = final_data.index.values.tolist()
final_data = final_data.reset_index()

# load the encodings of tweets previously save
loaded_encodings_flag_true = np.load('tweet_encodings_flag_true.npy')
encodings = loaded_encodings_flag_true[f_indexes]

# ## Preprocessing the comments to remove urls, user and subreddit mentions, punctuations and newline characters

import faiss
dimension = 768

def remove_url(post):
    result = re.sub(r"http\S+", "", post)
    return result

def remove_punctuations(post):
    result = post.translate(str.maketrans('', '', string.punctuation))
    return result

def remove_newlines(post):
    result = post.translate(str.maketrans('\n', ' '))
    return result

def remove_user_mentions(post):
    result = re.sub(r"/u/\S+", "", post)
    return result

def remove_subreddit_mentions(post):
    result = re.sub(r"/r/\S+", "", post)
    return result


# takes list of posts as input and returns list of pre-processed posts as output
''' 
Todo:
- Remove emojis
- check if the body says [deleted]
- check if the body is null
'''
def preprocess(posts):
    processed_posts = []
    for post in posts:
        result = remove_newlines(remove_punctuations(
            remove_subreddit_mentions(remove_user_mentions(remove_url(post)))))
        processed_posts.append(result)
    return processed_posts

def normalize_rows(x: np.ndarray):
    return x/np.linalg.norm(x, ord=2, axis=1, keepdims=True)

res = faiss.StandardGpuResources()
index_true_flatIP = faiss.IndexFlatIP(dimension)
gpu_index = faiss.index_cpu_to_all_gpus(index_true_flatIP)
gpu_index.add(normalize_rows(encodings))    

# normalizes the vectors in ndarray row wise
def normalize_rows(x: np.ndarray):
    return x/np.linalg.norm(x, ord=2, axis=1, keepdims=True)
print('before initializing Bert client')
# Instantiate the BERT client with the IP address logged in server.log
from bert_serving.client import BertClient
new_bc = BertClient(ip = '139.19.15.76', check_length=False)
print('After initializing Bert Client')
'''
Encodes the reddit comments for the month 06.2016 and
stores the similarity scores and corresponding index 
in RC_2016-06_scores.txt
'''
filename = '/INET/state-trolls/work/state-trolls/reddit_dataset/comments/RC_2016-06.bz2.decompressed'
output_filename = '/INET/state-trolls/work/state-trolls/reddit_dataset/comments/RC_2016-06_scores.txt'

# Reads 2560 lines of reddit data every iteration and creates a pandas dataframe for that chunk
for chunk in pd.read_json(filename,lines = True, chunksize=2560):
        print("In for loop")
        # preprocess the posts
        posts = chunk['body']
        posts = preprocess(posts)
        chunk = chunk.assign(preprocessed_body = posts)
        chunk_id_body = chunk[['id', 'preprocessed_body']]
        is_not_deleted_body = chunk_id_body['preprocessed_body'].apply(lambda x: not ('deleted' == x))
        chunk_id_body = chunk_id_body[is_not_deleted_body]
        # removes the entries having just space after preprocessing
        is_not_empty_string = chunk_id_body['preprocessed_body'].apply(lambda x: not str.isspace(x))
        chunk_id_body = chunk_id_body[is_not_empty_string]
        # removes the entries having empty string after preprocessing
        is_not_empty_string = chunk_id_body['preprocessed_body'].apply(lambda x: not x == '')
        chunk_id_body = chunk_id_body[is_not_empty_string]
        # encode the comments
        encoding_start_time = time.perf_counter()
        encoded_comments = new_bc.encode(chunk_id_body['preprocessed_body'].to_list())
        encoding_end_time = time.perf_counter()
        query = normalize_rows(encoded_comments)
        start_time = time.perf_counter()
        D, I = gpu_index.search(query, 300) 
        end_time = time.perf_counter()
        print("Time taken for encoding: ", (encoding_end_time - encoding_start_time)/60.0 ," minutes")
        print("Time taken for index search: ", (end_time - start_time)/60.0 ," minutes")
        i = 0
        # store the results of faiss
        for post_id in chunk_id_body['id']:
            with open(output_filename, "ab") as myFile:
#                 print(D[i], I[i])
                pickle.dump({post_id:np.asarray([D[i],I[i]])}, myFile)
            i = i + 1

