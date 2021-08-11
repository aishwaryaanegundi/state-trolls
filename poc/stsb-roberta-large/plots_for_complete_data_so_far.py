import numpy as np 
import pandas as pd
import csv
import os
import re
import datetime
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
import collections
import glob as glob


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


def plot_cdf(list_counts, xlabel, path, leg=False, islogx=True, xlimit=False):
    t_col = "#235dba"
    g_col = "#005916"
    c_col = "#a50808"
    r_col = "#ff9900"
    black = "#000000"
    pink = "#f442f1"
    t_ls = '-'
    r_ls = '--'
    c_ls = ':'
    g_ls = '-.'

    markers = [".", "o", "v", "^", "<", ">", "1", "2"]
    colors = [t_col, c_col, g_col, r_col, black, 'c', 'm', pink]
    line_styles = [t_ls, r_ls, c_ls, g_ls,t_ls, r_ls, c_ls, g_ls, t_ls]
    colors = colors[1:]
    line_styles= line_styles[1:]
    while(len(list_counts) > len(colors)):
        colors = colors + shuffle(colors)
        line_styles = line_styles + shuffle(line_styles)
        
    if xlimit:
        l2 = []
        for l in list_counts:
            l2_1 = [x for x in l if x<=xlimit]
            l2.append(l2_1)
        list_counts = l2
    
    for l in list_counts:
        l.sort()
    fig, ax = plt.subplots(figsize=(6,4))
    yvals = []
    for l in list_counts:
        yvals.append(np.arange(len(l))/float(len(l)-1))
    for i in range(len(list_counts)):
        ax.plot(list_counts[i], yvals[i], color=colors[i], linestyle=line_styles[i])
    if islogx:
        ax.set_xscale("log")
    plt.xlabel(xlabel)
    plt.ylabel('CDF')
    plt.grid()
    for item in ([ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(13)
    
    if leg:
        plt.legend(leg, loc='best', fontsize=13)
    
    plt.show()
    fig.savefig(path, bbox_inches='tight')

# combined_annotation = pd.DataFrame()

# hits = pd.DataFrame()
# hits = hits.append(pd.read_csv('./annotations/2016.10.annotation_data_point_8.csv'), ignore_index = True)
# hits = hits.append(pd.read_csv('./annotations/2016.10.annotation_data_point_85.csv'), ignore_index = True)
# hits = hits.append(pd.read_csv('./annotations/2016.10.annotation_data_point_9.csv'), ignore_index = True)
# hits = hits.append(pd.read_csv('./annotations/2016.10.annotation_data_point_95.csv'), ignore_index = True)
# posts = pd.read_csv('/INET/state-trolls/work/state-trolls/reddit_dataset/comments/posts/posts-2016-10.csv')        
# times = []
# subreddits = []
# for idx, row in hits.iterrows():
#     element = posts[posts['id'] == row['post_id']]
#     time = element['created_utc'].astype('int64')
#     time = str(time).split()
# #     print(time)
#     try:
#         times.append(datetime.datetime.fromtimestamp(int(time[1])))
#         subreddits.append((element['subreddit'].values)[0])
#     except ValueError:
# #         print(time[0])
#         times.append(time[0])
#     except IndexError:
#         times.append('nil')
#         subreddit.append('nil')
                   
# hits['post_time'] = times
# hits['subreddit'] = subreddits
# combined_annotation = combined_annotation.append(hits, ignore_index = True)
# print('Added a months data')      
# hits = pd.DataFrame()
# hits = hits.append(pd.read_csv('./annotations/2016.11.annotation_data_point_8.csv'), ignore_index = True)
# hits = hits.append(pd.read_csv('./annotations/2016.11.annotation_data_point_85.csv'), ignore_index = True)
# hits = hits.append(pd.read_csv('./annotations/2016.11.annotation_data_point_9.csv'), ignore_index = True)
# hits = hits.append(pd.read_csv('./annotations/2016.11.annotation_data_point_95.csv'), ignore_index = True)
# posts = pd.read_csv('/INET/state-trolls/work/state-trolls/reddit_dataset/comments/posts/posts-2016-11.csv')        
# times = []
# subreddits = []
# for idx, row in hits.iterrows():
#     element = posts[posts['id'] == row['post_id']]
#     time = element['created_utc'].astype('int64')
#     time = str(time).split()
# #     print(time)
#     try:
#         times.append(datetime.datetime.fromtimestamp(int(time[1])))
#         subreddits.append((element['subreddit'].values)[0])
#     except ValueError:
#         print(time[0])
#         times.append(time[0])
#     except IndexError:
#         times.append('nil')
#         subreddit.append('nil')
                   
# hits['post_time'] = times
# hits['subreddit'] = subreddits
# combined_annotation = combined_annotation.append(hits, ignore_index = True)
# print('Added a months data')      
                   
# hits = pd.DataFrame()
# hits = hits.append(pd.read_csv('./annotations/2017.04.annotation_data_point_8.csv'), ignore_index = True)
# hits = hits.append(pd.read_csv('./annotations/2017.04.annotation_data_point_85.csv'), ignore_index = True)
# hits = hits.append(pd.read_csv('./annotations/2017.04.annotation_data_point_9.csv'), ignore_index = True)
# hits = hits.append(pd.read_csv('./annotations/2017.04.annotation_data_point_95.csv'), ignore_index = True)
# posts = pd.read_csv('/INET/state-trolls/work/state-trolls/reddit_dataset/comments/posts/posts-2017-04.csv')        
# times = []
# subreddits = []
# for idx, row in hits.iterrows():
#     element = posts[posts['id'] == row['post_id']]
#     time = element['created_utc'].astype('int64')
#     time = str(time).split()
# #     print(time)
#     try:
#         times.append(datetime.datetime.fromtimestamp(int(time[1])))
#         subreddits.append((element['subreddit'].values)[0])
#     except ValueError:
#         print(time[0])
#         times.append(time[0])
#     except IndexError:
#         times.append('nil')
#         subreddit.append('nil')
        
# hits['post_time'] = times
# hits['subreddit'] = subreddits
# combined_annotation = combined_annotation.append(hits, ignore_index = True) 
# print('Added a months data')      
                   
# hits = pd.read_csv('/INET/state-trolls/work/state-trolls/reddit_dataset/comments/hits/hits_2016-02.csv')
# posts = pd.read_csv('/INET/state-trolls/work/state-trolls/reddit_dataset/comments/posts/posts-2016-02.csv')        
# times = []
# subreddits = []
# for idx, row in hits.iterrows():
#     element = posts[posts['id'] == row['post_id']]
#     time = element['created_utc'].astype('int64')
#     time = str(time).split()
# #     print(time)
#     try:
#         times.append(datetime.datetime.fromtimestamp(int(time[1])))
#         subreddits.append((element['subreddit'].values)[0])
#     except ValueError:
#         print(time[0])
#         times.append(time[0])
#     except IndexError:
#         times.append('nil')
#         subreddit.append('nil')
                   
# hits['post_time'] = times
# hits['subreddit'] = subreddits
# combined_annotation = combined_annotation.append(hits, ignore_index = True)
# print('Added a months data')      
                   
# hits = pd.read_csv('/INET/state-trolls/work/state-trolls/reddit_dataset/comments/hits/hits_2016-03.csv')
# posts = pd.read_csv('/INET/state-trolls/work/state-trolls/reddit_dataset/comments/posts/posts-2016-03.csv')        
# times = []
# subreddits = []
# for idx, row in hits.iterrows():
#     element = posts[posts['id'] == row['post_id']]
#     time = element['created_utc'].astype('int64')
#     time = str(time).split()
# #     print(time)
#     try:
#         times.append(datetime.datetime.fromtimestamp(int(time[1])))
#         subreddits.append((element['subreddit'].values)[0])
#     except ValueError:
#         print(time[0])
#         times.append(time[0])
#     except IndexError:
#         times.append('nil')
#         subreddit.append('nil')
                   
# hits['post_time'] = times
# hits['subreddit'] = subreddits
# combined_annotation = combined_annotation.append(hits, ignore_index = True)      
# print('Added a months data')      
                   
# hits = pd.read_csv('/INET/state-trolls/work/state-trolls/reddit_dataset/comments/hits/hits_2016-04.csv')
# posts = pd.read_csv('/INET/state-trolls/work/state-trolls/reddit_dataset/comments/posts/posts-2016-04.csv')        
# times = []
# subreddits = []
# for idx, row in hits.iterrows():
#     element = posts[posts['id'] == row['post_id']]
#     time = element['created_utc'].astype('int64')
#     time = str(time).split()
# #     print(time)
#     try:
#         times.append(datetime.datetime.fromtimestamp(int(time[1])))
#         subreddits.append((element['subreddit'].values)[0])
#     except ValueError:
#         print(time[0])
#         times.append(time[0])
#     except IndexError:
#         times.append('nil')
#         subreddit.append('nil')
                   
# hits['post_time'] = times
# hits['subreddit'] = subreddits
# combined_annotation = combined_annotation.append(hits, ignore_index = True)
# print('Added a months data')      
                   
# hits = pd.read_csv('/INET/state-trolls/work/state-trolls/reddit_dataset/comments/hits/hits_2016-05.csv')
# posts = pd.read_csv('/INET/state-trolls/work/state-trolls/reddit_dataset/comments/posts/posts-2016-05.csv')        
# times = []
# subreddits = []
# for idx, row in hits.iterrows():
#     element = posts[posts['id'] == row['post_id']]
#     time = element['created_utc'].astype('int64')
#     time = str(time).split()
# #     print(time)
#     try:
#         times.append(datetime.datetime.fromtimestamp(int(time[1])))
#         subreddits.append((element['subreddit'].values)[0])
#     except ValueError:
#         print(time[0])
#         times.append(time[0])
#     except IndexError:
#         times.append('nil')
#         subreddit.append('nil')
                   
# hits['post_time'] = times
# hits['subreddit'] = subreddits
# combined_annotation = combined_annotation.append(hits, ignore_index = True)
# print('Added a months data')                         
# print('Number of hits gathered :', combined_annotation.shape)

# fdata = data[data['tweetid'].isin(combined_annotation['tweet_id'].to_list())]
# times = []
# for idx, row in combined_annotation.iterrows():
#     element = fdata[fdata['tweetid'] == row['tweet_id']]
#     time = element['tweet_time'].to_string()
#     time = time.split()
# #     print(time[1]+' '+time[2])
#     times.append(time[1]+' '+time[2])
            
# combined_annotation['tweet_time'] = times
# combined_annotation.to_csv('./combined_data.csv')
# combined_annotation = combined_annotation[combined_annotation['post_time'] !=  'Series([],']
# combined_annotation = combined_annotation[combined_annotation['post_time'] !=  'nil']
# t = pd.to_datetime(combined_annotation.tweet_time, infer_datetime_format=True)                   
# del combined_annotation['tweet_time']
# combined_annotation['tweet_time'] = t
# t = pd.to_datetime(combined_annotation.post_time, infer_datetime_format=True)
# del combined_annotation['post_time']
# combined_annotation['post_time'] = t 
# td = (combined_annotation.post_time - combined_annotation.tweet_time).astype('timedelta64[h]')
# combined_annotation['time_difference'] = td
# combined_annotation.to_csv('./combined_data_td.csv')        


### New faster approach
# Collect all hits above 0.8 cosine similarity
months = ['2016-02','2016-03','2016-04','2016-05','2016-10','2016-11','2017-04']

def get_relevant_hits(month):
    combined_hits = pd.DataFrame()
    reddit_data = pd.read_csv('/INET/state-trolls/work/state-trolls/reddit_dataset/comments/posts/posts-'+ month +'.csv')
    for f in glob.glob('../../reddit_dataset/comments/scores/RC_'+ month +'.bz2.decompressed/*.txt'):
        hits = []
        count = 0
        with open(f , 'r') as content_file:
            content = content_file.read()
            json_data = content.replace('][',',')
            j_object = json.loads(json_data)
            json_df = pd.DataFrame(j_object)
            j_df = json_df[(json_df['cosine_similarity'] > 0.8)]
            fdata = data[data['tweetid'].isin(j_df['tweet_id'].to_list())]
            for index, row in j_df.iterrows():
                post = reddit_data[reddit_data['id'] == row['post_id']]
                if not (post.empty):
                    try:
                        tweet = fdata[fdata['tweetid'] == row['tweet_id']]
                        post_time = (post['created_utc'].values)[0]
                        post_time = (datetime.datetime.fromtimestamp(post_time))
                        element = fdata[fdata['tweetid'] == row['tweet_id']]
                        tweet_time = (element['tweet_time'].values)[0]
                        hits.append({'tweet_id': row['tweet_id'], 'post_id': row['post_id'], 
                                     'post_author': (post['author'].values)[0],
                                     'post_time': post_time, 'tweet_time': tweet_time})
                    except IndexError:
                        print(post)
                        print(post['created_utc'])
                    except ValueError:
                        print(post)
                        print(post_time)
                    except Exception as ex:
                        print(post)
                        print(ex)
                        
            hits = pd.DataFrame(hits)
            if not (hits.empty):
                t = pd.to_datetime(hits.tweet_time, infer_datetime_format = True)
                del hits['tweet_time']
                hits['tweet_time'] = t
                hits['td'] = (hits.post_time - hits.tweet_time).astype('timedelta64[h]')
                hits = hits[(hits['td'] >= -672) & (hits['td'] <= 672)]
        combined_hits.append(hits, ignore_index = True)
        count = count + 1
        print(count)
    return combined_hits

def save_data(name, data):
    data.to_csv('./results/group_by_author/'+name+'.csv')
    
def get_unique_elements_count(data, field):
    counts = []
    author_wise_lists = data[field].tolist()
    for author_posts in author_wise_lists():
        concatenated_list = []
        for posts in author_posts:
            concatenated_list = np.concatenate([concatenated_list, posts], axis=0)
        unique_elements = np.unique(concatenated_list)
        counts.append(len(unique_elements))
    return counts
   
authors_avg_time_difference = pd.DataFrame()
ow_hits_per_author = pd.DataFrame()
ow_posts_per_author = pd.DataFrame()
ow_tweets_per_author = pd.DataFrame()

tw_hits_per_author = pd.DataFrame()
tw_posts_per_author = pd.DataFrame()
tw_tweets_per_author = pd.DataFrame()

om_hits_per_author = pd.DataFrame()
om_posts_per_author = pd.DataFrame()
om_tweets_per_author = pd.DataFrame()


for m in months:
    relevant_hits = get_relevant_hits(m)
    print('Number of hits in month ', m, 'with time difference less than a month :', (relevant_hits.shape)[0])
    
    owh = relevant_hits[(relevant_hits['td'] >= -168) & (relevant_hits['td'] <= 168)]
    hits_per_author = owh.groupby(['post_author'])["post_id"].count().reset_index(name="count")
    ow_hits_per_author = ow_hits_per_author.append(hits_per_author, ignore_index = True)
    avg_td = owh.groupby(['post_author'])["td"].mean().reset_index(name="average")
    authors_avg_time_difference = authors_avg_time_difference.append(avg_td, ignore_index = True)
    posts_per_author = owh.groupby(['post_author'])["post_id"].unique().reset_index(name="unique_post_ids")
    ow_posts_per_author = ow_posts_per_author.append(posts_per_author, ignore_index = True)
    tweets_per_author = owh.groupby(['post_author'])["tweet_id"].unique().reset_index(name="unique_tweet_ids")
    ow_tweets_per_author = ow_tweets_per_author.append(tweets_per_author, ignore_index = True)
    
    twh = relevant_hits[(relevant_hits['td'] >= -336) & (relevant_hits['td'] <= 336)]
    hits_per_author = twh.groupby(['post_author'])["post_id"].count().reset_index(name="count")
    tw_hits_per_author = tw_hits_per_author.append(hits_per_author, ignore_index = True)
    posts_per_author = twh.groupby(['post_author'])["post_id"].unique().reset_index(name="unique_post_ids")
    tw_posts_per_author = tw_posts_per_author.append(posts_per_author, ignore_index = True)
    tweets_per_author = twh.groupby(['post_author'])["tweet_id"].unique().reset_index(name="unique_tweet_ids")
    tw_tweets_per_author = tw_tweets_per_author.append(tweets_per_author, ignore_index = True)
    
    hits_per_author = relevant_hits.groupby(['post_author'])["post_id"].count().reset_index(name="count")
    om_hits_per_author = om_hits_per_author.append(hits_per_author, ignore_index = True)
    posts_per_author = relevant_hits.groupby(['post_author'])["post_id"].unique().reset_index(name="unique_post_ids")
    om_posts_per_author = om_posts_per_author.append(posts_per_author, ignore_index = True)
    tweets_per_author = relevant_hits.groupby(['post_author'])["tweet_id"].unique().reset_index(name="unique_tweet_ids")
    om_tweets_per_author = om_tweets_per_author.append(tweets_per_author, ignore_index = True)
    
    save_data('authors_avg_time_difference', authors_avg_time_difference)
    save_data('ow_hits_per_author', ow_hits_per_author)
    save_data('ow_posts_per_author', ow_posts_per_author)
    save_data('ow_tweets_per_author', ow_tweets_per_author)
    save_data('tw_hits_per_author', tw_hits_per_author)
    save_data('tw_posts_per_author', tw_posts_per_author)
    save_data('tw_tweets_per_author', tw_tweets_per_author)
    save_data('om_hits_per_author', om_hits_per_author)
    save_data('om_posts_per_author', om_posts_per_author)
    save_data('om_tweets_per_author', om_tweets_per_author)
    
avg_td = authors_avg_time_difference.groupby(['post_author'])["average"].mean().reset_index(name="average")

ow_hits = ow_hits_per_author.groupby(['post_author'])["count"].count().reset_index(name="count")
tw_hits = tw_hits_per_author.groupby(['post_author'])["count"].count().reset_index(name="count")
om_hits = om_hits_per_author.groupby(['post_author'])["count"].count().reset_index(name="count")

ow_posts = get_unique_elements_count(ow_posts_per_author
                                     .groupby('post_author')
                                     .agg({'unique_post_ids':list})
                                     .reset_index(), 'unique_post_ids')
tw_posts = get_unique_elements_count(tw_posts_per_author
                                     .groupby('post_author')
                                     .agg({'unique_post_ids':list})
                                     .reset_index(), 'unique_post_ids')
om_posts = get_unique_elements_count(om_posts_per_author
                                     .groupby('post_author')
                                     .agg({'unique_post_ids':list})
                                     .reset_index(), 'unique_post_ids')

ow_tweets = get_unique_elements_count(ow_tweets_per_author
                                     .groupby('post_author')
                                     .agg({'unique_tweet_ids':list})
                                     .reset_index(), 'unique_tweet_ids')
tw_tweets = get_unique_elements_count(tw_tweets_per_author
                                     .groupby('post_author')
                                     .agg({'unique_tweet_ids':list})
                                     .reset_index(), 'unique_tweet_ids')
om_tweets = get_unique_elements_count(om_tweets_per_author
                                     .groupby('post_author')
                                     .agg({'unique_tweet_ids':list})
                                     .reset_index(), 'unique_tweet_ids')

plot_cdf([avg_td], 'hits/author',leg=['1 week time difference'], path = './results/group_by_author/cdf_of_avgtd_per_author_for_one_week_td.pdf', islogx=False)

plot_cdf([ow_hits, tw_hits, om_hits], 'hits/author',leg=['1 week time difference', '2 week time difference', '1 month time difference'], path = './results/group_by_author/cdf_of_hits_per_author_for_varying_td.pdf', islogx=True)

plot_cdf([ow_posts, tw_posts, om_posts], 'posts/author',leg=['1 week time difference', '2 week time difference', '1 month time difference'], path = './results/group_by_author/cdf_of_posts_per_author_for_varying_td.pdf', islogx=True)

plot_cdf([ow_tweets, tw_tweets, om_tweets], 'tweets/author',leg=['1 week time difference', '2 week time difference', '1 month time difference'], path = './results/group_by_author/cdf_of_tweets_per_author_for_varying_td.pdf', islogx=True)
