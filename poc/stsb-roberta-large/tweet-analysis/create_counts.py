import numpy as np 
import pandas as pd
import string
import json
import glob
import os
import collections
import time

i = 0 
final_df = pd.DataFrame(columns = ['tweet_id', 'frequency'])
for name in glob.glob('/INET/state-trolls/work/state-trolls/reddit_dataset/comments/scores/RC_2016-01.bz2.decompressed/*'):
    s = time.perf_counter()
    with open(name , 'r') as content_file:
        content = content_file.read()
        json_data = content.replace('][',',')
        j_object = json.loads(json_data)
        j_df = pd.DataFrame(j_object)
        counts = collections.Counter(j_df['tweet_id'].tolist())
        counts_sorted = counts.most_common()
        c_df = pd.DataFrame(counts_sorted, columns=['tweet_id', 'frequency'])
        final_df = final_df.append(c_df, ignore_index = True) 
        final_df['t_f'] = final_df.groupby(['tweet_id'])['frequency'].transform('sum')
        final_df = final_df.drop_duplicates(subset=['tweet_id'])
        del final_df['frequency']
        final_df.columns = ['tweet_id', 'frequency']
#         print(final_df.head())
        print(len(final_df))
        i = i + 1
        if ((i % 10 ) == 0):
            final_df.to_csv('counts-10.csv') 
        e = time.perf_counter()
        print('time taken for one iteration: ', i, '  ',  e-s/60.0)
            
final_df.to_csv('counts-10.csv')