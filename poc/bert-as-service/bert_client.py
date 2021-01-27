import numpy as np 
import pandas as pd
import csv
import os
import re
import emoji
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import pickle
import random
import string
import json

from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine

# displays all columns and rows when asked to print
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', -1)

# obtain the filename of the reddit data to be encoded
import sys, getopt
inputfile = ''
def main(argv):
    global inputfile
    try:
        opts, args = getopt.getopt(argv,"hi:o",["ifile=","ofile="])
    except getopt.GetoptError:
      print ('test.py -i <inputfile> -o <outputfile>')
      sys.exit(2)
    for opt, arg in opts:
      if opt == '-h':
         print ('test.py -i <inputfile> -o <outputfile>')
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputfile = arg
      elif opt in ("-o", "--ofile"):
         outputfile = arg
    print('Input file is ', inputfile)
    

if __name__ == "__main__":
   main(sys.argv[1:])
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

def remove_url(tweet):
    result = re.sub(r"http\S+", "", tweet)
    return result

def remove_mentions(tweet):
    result = re.sub(r"@\S+", "", tweet)
    return result

def remove_retweet(tweet):
    result = re.sub(r"RT @\S+", "", tweet)
    return result

def remove_punctuations(tweet):
    result = tweet.translate(str.maketrans('', '', string.punctuation))
    return result

def remove_emojis(tweet):
    result = re.sub(emoji.get_emoji_regexp(), "", tweet)
    return result

def remove_special_char(tweet):
    result = re.sub(r"[^a-zA-Z0-9 ]", "", tweet)
    return result

# takes list of tweets as input and returns list of pre-processed tweets as output
def preprocess(tweets):
    processed_tweets = []
    for tweet in tweets:
        result = remove_special_char(remove_punctuations(remove_emojis(remove_mentions(remove_retweet(remove_url(tweet))))))
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

# obtain the encodings for the filtered data
from bert_serving.client import BertClient
# bc = BertClient(ip='139.19.15.55')
# encodings = bc.encode(f_english_tweet_data['processed_tweets'].to_list())
# print("Number of dimensions in the encodings: ",encodings.shape[1])

# save the encodings for later use. Order preserved
# np.save('encodings_for_filtered_data', encodings)
encodings = np.load('encodings_for_removed_emojis.npy')
print('shape of the encoding: ', encodings.shape)

# ## Preprocessing the comments to remove urls, user and subreddit mentions, punctuations and newline characters

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

def remove_emojis(post):
    result = re.sub(emoji.get_emoji_regexp(), "", post)
    return result

def remove_special_char(post):
    result = re.sub(r"[^a-zA-Z0-9 ]", "", post)
    return result


# takes list of posts as input and returns list of pre-processed posts as output
''' 
Todo:
- check if the body says [deleted]
- check if the body is null
'''
def preprocess(posts):
    processed_posts = []
    for post in posts:
        result = remove_newlines(remove_special_char(remove_emojis(remove_punctuations(
            remove_subreddit_mentions(remove_user_mentions(remove_url(post)))))))
        processed_posts.append(result)
    return processed_posts

# normalizes the vectors in ndarray row wise
def normalize_rows(x: np.ndarray):
    return x/np.linalg.norm(x, ord=2, axis=1, keepdims=True)

# create faiss index with obtained encodings
import faiss
dimension = 768
res = faiss.StandardGpuResources()
index_true_flatIP = faiss.IndexFlatIP(dimension)
gpu_index = faiss.index_cpu_to_all_gpus(index_true_flatIP)
print('normalized rows shape:', (normalize_rows(encodings)).shape)
gpu_index.add(normalize_rows(encodings))    

# Get the IP address
with open('server_ip_address.txt', "r") as f:
    server_ip_address = (f.readline()).strip()
print('server ip address: ', server_ip_address)
# BERT client to encode reddit posts
new_bc = BertClient(ip = server_ip_address, check_length=False)

# Get the iteration number
with open("json_read_iteration.txt", "r") as file1: 
    iterated = int(file1.readline())
    
# input and output file definition
filename = '/INET/state-trolls/work/state-trolls/reddit_dataset/comments/' + inputfile
output_filename = '/INET/state-trolls/work/state-trolls/reddit_dataset/comments/' + inputfile + '-' + str(iterated)+'_scores_emojis.json'
print('starting from iteration: ', iterated)
iteration = 0
for chunk in pd.read_json(filename,lines = True, chunksize=10000):
    if ((iteration >= iterated) & (iteration < iterated + 2500)):
        start_time = time.perf_counter()
#         output_filename = '/INET/state-trolls/work/state-trolls/reddit_dataset/comments/' + inputfile + '-' + str(iteration)+'_scores.json'
        with open(output_filename, 'w') as to_file:
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
            encoded_comments = new_bc.encode(chunk_id_body['preprocessed_body'].to_list())
            query = normalize_rows(encoded_comments)
    #         start_time = time.perf_counter()
            D, I = gpu_index.search(query, 1000) 
    #         end_time = time.perf_counter()
    #         print("Time taken for index search: ", (end_time - start_time)/60.0 ," minutes")
            i = 0
            for post_id in chunk_id_body['id']:
                scores = D[i]
                indices = I[i]
    #             last_index = 999
                for idx, score in enumerate(scores):
                  if score < 0.95:
                      break
                  tweet_idx = indices[idx]
                  cos_sim = scores[idx]
                  record = {'tweet_id':str(f_english_tweet_data.iloc[tweet_idx]['tweetid']), 
                            'post_id':post_id, 'cosine_similarity': str(cos_sim)}
                  json.dump(record, to_file)      
    #                     last_index = idx
                i = i + 1
        end_time = time.perf_counter()
        print("Time taken for iteration ", iteration, 'is : ', (end_time - start_time)/60.0 ," minutes")
    elif not (iteration < iterated + 2500):
        break
    iteration = iteration + 1
print('iteration to be written to file: ', iteration)
with open("json_read_iteration.txt", "w") as file1: 
     file1.write(str(iteration))
        
if (iteration < iterated + 2500):
    with open("job_status.txt", "w") as file1: 
        file1.write(str('1'))
    