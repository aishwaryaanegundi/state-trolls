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
dataset_paths = ["../../datasets/russia_052020_tweets_csv_hashed_2.csv", 
         "../../datasets/russian_linked_tweets_csv_hashed.csv", 
         "../../datasets/ira_tweets_csv_hashed.csv", 
         "../../datasets/russia_201906_1_tweets_csv_hashed.csv"]

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
    
    
from sentence_transformers import SentenceTransformer, LoggingHandler
import logging

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


if __name__ == '__main__':
    main(sys.argv[1:])
    #Create a large list of 100k sentences
    sentences = f_english_tweet_data['processed_tweets'].to_list()
    #Define the model
    model = SentenceTransformer('stsb-roberta-large')
    model.max_seq_length = 300
    #Start the multi-process pool on all available CUDA devices
    pool = model.start_multi_process_pool()
    #Compute the embeddings using the multi-process pool
#     start_time = time.perf_counter()
#     print('Beginning to encode\n')
#     encodings = model.encode_multi_process(sentences, pool)
#     end_time = time.perf_counter()
#     print("Time taken for encoding is: ", (end_time - start_time)/60.0 ," minutes")
#     print("Number of dimensions in the encodings", encodings.shape)
    # save the encodings for later use. Order preserved
#     np.save('encodings_stsb_roberta_large', encodings)
    encodings = np.load('encodings_stsb_roberta_large.npy')
    print('shape of the encoding after reloading: ', encodings.shape)
#     model.stop_multi_process_pool(pool)
  

    # takes list of posts as input and returns list of pre-processed posts as output
    ''' 
    Todo:
    - check if the body says [deleted]
    - check if the body is null
    '''
    def preprocess(posts):
        processed_posts = []
        for post in posts:
            result = re.sub(r"http\S+", "", post)
            result = re.sub(r"/u/\S+", "", result)
            result = re.sub(r"/r/\S+", "", result)
            result = re.sub(emoji.get_emoji_regexp(), "", result)
            result_removed_punctuation = result.translate(str.maketrans('', '', string.punctuation))
            result = re.sub(r"[^a-zA-Z0-9 ]", "", result_removed_punctuation)
            result_removed_newlines = result.translate(str.maketrans('\n', ' '))
            processed_posts.append(result_removed_newlines)
        return processed_posts

    # normalizes the vectors in ndarray row wise
    def normalize_rows(x: np.ndarray):
        return x/np.linalg.norm(x, ord=2, axis=1, keepdims=True)

    # create faiss index with obtained encodings
    import faiss
    dimension = 1024
    res = faiss.StandardGpuResources()
    index_true_flatIP = faiss.IndexFlatIP(dimension)
    gpu_index = faiss.index_cpu_to_all_gpus(index_true_flatIP)
    print('normalized rows shape:', (normalize_rows(encodings)).shape)
    gpu_index.add(normalize_rows(encodings))    

    # Get the iteration number
    with open("json_read_iteration.txt", "r") as file1: 
        iterated = int(file1.readline())

    # input and output file definition
    filename = '/INET/state-trolls/work/state-trolls/reddit_dataset/comments/' + inputfile
    output_filename = '/INET/state-trolls/work/state-trolls/reddit_dataset/comments/' + inputfile + '-' + str(iterated)+'_scores_stsb.json'
    print('starting from iteration: ', iterated)
    iteration = 0
    for chunk in pd.read_json(filename,lines = True, chunksize=100):
        print('Length of chunk before removing automoderator posts: ', len(chunk))
        chunk = chunk[chunk.author != 'AutoModerator']
        print('Length of chunk after removing automoderator posts: ', len(chunk))
        id_sentences = []
        if ((iteration >= iterated) & (iteration < iterated + 2)):
            start_time = time.perf_counter()
            with open(output_filename, 'w') as to_file:
                sent_tokenize_begin = time.perf_counter()
                for index, row in chunk.iterrows():
                    sentences = nltk.sent_tokenize(row['body'])                
                    for idx,sentence in enumerate(sentences):
                        id_sentences.append({'id':row['id'], 'body': sentence, 's_id': idx})
                sent_tokenize_end = time.perf_counter()
                print("Time taken for sentence tokenization is :",
                      (sent_tokenize_end - sent_tokenize_begin)/60.0 ," minutes")
                print('length of sentences is: ', len(id_sentences))
                sentence_df = pd.DataFrame(id_sentences)
                sentence_texts = sentence_df['body']
                sentence_texts = preprocess(sentence_texts)
                sentence_df = sentence_df.assign(preprocessed_body = sentence_texts)
                chunk = sentence_df
                chunk_id_body = chunk[['id', 'preprocessed_body','s_id']]
                is_not_deleted_body = chunk_id_body['preprocessed_body'].apply(lambda x: not ('deleted' == x))
                chunk_id_body = chunk_id_body[is_not_deleted_body]
                # removes the entries having just space after preprocessing
                is_not_empty_string = chunk_id_body['preprocessed_body'].apply(lambda x: not str.isspace(x))
                chunk_id_body = chunk_id_body[is_not_empty_string]
                # removes the entries having empty string after preprocessing
                is_not_empty_string = chunk_id_body['preprocessed_body'].apply(lambda x: not x == '')
                chunk_id_body = chunk_id_body[is_not_empty_string]
                # encode the comments
                start_time_e = time.perf_counter()
                encoded_comments = model.encode_multi_process(chunk_id_body['preprocessed_body'].to_list(),
                                                              pool,batch_size = 128)
                end_time_e = time.perf_counter()
                print("Time taken for encoding comments is :", (end_time_e - start_time_e)/60.0 ,
                      " minutes. Iteration: ", iteration)
                query = normalize_rows(encoded_comments)
#                 start_time_s = time.perf_counter()
#                 D, I = gpu_index.search(query, 1000) 
#                 end_time_s = time.perf_counter()
#                 print("Time taken for index search: ", (end_time_s - start_time_s)/60.0 ," minutes")
                i = 0
                for loc, entry in chunk_id_body.iterrows():
                    q = query[i]
                    start_time_s = time.perf_counter()
                    D, I = gpu_index.search(q, 1662390) 
                    end_time_s = time.perf_counter()
                    print("Time taken for index search: ", (end_time_s - start_time_s)/60.0 ," minutes")
                    scores = D[0]
                    indices = I[0]
                    o_f = '/INET/state-trolls/work/state-trolls/reddit_dataset/comments/annotations/' + inputfile + '-' + str(iterated)+str(i)+'.json'
                    with open(o_f, 'w') as t_f:
                        for idx, score in enumerate(scores):
                          tweet_idx = indices[idx]
                          cos_sim = scores[idx]
                          record = {'tweet_id':str(f_english_tweet_data.iloc[tweet_idx]['tweetid']), 
                                    'post_id':entry['id'],'s_id':entry['s_id'], 'cosine_similarity': str(cos_sim)}
                          json.dump(record, t_f)      
                    i = i + 1
                    print('completed query: ', i)
            end_time = time.perf_counter()
            print("Time taken for iteration ", iteration, 'is : ', (end_time - start_time)/60.0 ," minutes")
        elif not (iteration < iterated + 2):
            break
        iteration = iteration + 1
    print('iteration to be written to file: ', iteration)
    with open("json_read_iteration.txt", "w") as file1: 
         file1.write(str(iteration))

    if (iteration < iterated + 2):
        with open("job_status.txt", "w") as file1: 
            file1.write(str('1'))




