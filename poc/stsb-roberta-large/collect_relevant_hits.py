import json
import pandas as pd
import glob
import nltk
import re
import emoji
import string

def save_data(d1):
    pd.DataFrame(d1).to_csv('/INET/state-trolls/work/state-trolls/reddit_dataset/comments/hits/hits_2016-03.csv')    
    
reddit_data = pd.read_csv('/INET/state-trolls/work/state-trolls/reddit_dataset/comments/posts/posts-2016-03.csv')
# reddit_data['created_utc'] = reddit_data['created_utc'].fillna(0)
# reddit_data['time'] = reddit_data['created_utc'].astype('int64')
twitter_data = pd.read_csv('./relevant_tweets.csv')

hits = []
count = 0
iteration = 0 
pd.set_option('max_colwidth', -1)
for f in glob.glob('../../reddit_dataset/comments/scores/RC_2016-03.bz2.decompressed/*.txt'):
    iteration = iteration + 1
    with open(f , 'r') as content_file:
        content = content_file.read()
        json_data = content.replace('][',',')
        j_object = json.loads(json_data)
        json_df = pd.DataFrame(j_object)
        
        j_df = json_df[(json_df['cosine_similarity'] > 0.8) & (json_df['cosine_similarity'] <= 1.0)]
        for index, row in j_df.iterrows():
            post = reddit_data[reddit_data['id'] == row['post_id']]
            sentences = nltk.sent_tokenize(post['body'].to_string())
            try:
                sentence = sentences[row['sent_id']]
                tweet = twitter_data[twitter_data['tweetid'] == row['tweet_id']]
                hits.append({'tweet_id': row['tweet_id'],
                                                 'tweet_text': tweet['tweet_text'].to_string(),
                                                 'post_id': row['post_id'],
                                                 'matching_sentence': sentence,
                                                'post_text' : post['body'].to_string(),
                                                'sim': row['cosine_similarity'],
                                                'post_author': post['author'].to_string()})
            except:
                count = count + 1
        
        print(iteration)
        if (iteration%10 == 0):
            save_data(hits)
                     
save_data(hits)     