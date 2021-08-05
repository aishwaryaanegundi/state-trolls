import glob
import pandas as pd
import json

def get_all_ids(directory):
#     count = 0
    all_ids = []
    for f in glob.glob('/INET/state-trolls/work/state-trolls/reddit_dataset/comments/scores/'+ directory +'/*txt'):
        try:
            with open(f , 'r') as content_file:
                content = content_file.read()
                json_data = content.replace('][',',')
                j_object = json.loads(json_data)
                j_df = pd.DataFrame(j_object)
                ids = j_df['post_id'].unique()
                for i in ids:
#                     if i not in all_ids:
                    all_ids.append(i)
#             count = count + 1
#             print(count)
        except:
            pass
    return list(set(all_ids))

ids = get_all_ids('RC_2016-12.bz2.decompressed')
print('length of the unique ids with atleast one hit: ', len(ids))

data = pd.read_csv('./relevant_tweets.csv')

posts = pd.DataFrame()
count = 0
for chunk in pd.read_json('/INET/state-trolls/work/state-trolls/reddit_dataset/comments/RC_2016-12.bz2.decompressed',
                          lines = True, chunksize=10000):
    chunk = chunk[chunk.author != 'AutoModerator']
    selective_chunk = chunk[chunk['id'].isin(ids)]
    print('selected chunk shape: ', selective_chunk.shape)
    posts = posts.append(selective_chunk, ignore_index = True)
    print('posts shape: ', posts.shape)
    count = count +1
    if((count%1000) == 0 ) :
        print(count)
        posts.to_csv('/INET/state-trolls/work/state-trolls/reddit_dataset/comments/posts/posts-2016-12.csv')
    posts.to_csv('/INET/state-trolls/work/state-trolls/reddit_dataset/comments/posts/posts-2016-12.csv')