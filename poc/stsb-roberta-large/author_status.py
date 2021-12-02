import pandas as pd
import requests

months = ['2016-01','2016-02','2016-03','2016-04','2016-05','2016-06',
         '2016-07','2016-08','2016-09','2016-10','2016-11','2016-12',
         '2017-01','2017-02','2017-03','2017-04','2017-05','2017-06',]

complete_hits = pd.DataFrame()

for m in months:
    relevant_hits = pd.read_csv('./results/relevent_hits/'+m+'.csv')
    complete_hits = complete_hits.append(relevant_hits, ignore_index = True)
    print('Number of hits in month ', m, 'with time difference less than a month :', (relevant_hits.shape)[0])
    
del complete_hits['Unnamed: 0']
del complete_hits['bot_author']

title_tweetids_df = pd.read_csv('./results/tweet_ids_of_news_titles_tweets.csv')
title_tweetids = title_tweetids_df['tweetid'].tolist()
print('complete hits before removing news titles: ', complete_hits.shape)
complete_hits_without_news = complete_hits[~complete_hits['tweet_id'].isin(title_tweetids)]
print('complete hits before after news titles: ', complete_hits_without_news.shape)
# Remove authors with name as bot or auto
def is_bot_auto(row):
    author = str(row['post_author'])
    if 'bot' in author:
        return True
    elif 'auto' in author:
        return True
    else:
        return False
complete_hits_without_news['is_bot'] = complete_hits_without_news.apply(is_bot_auto, axis=1)
complete_hits = complete_hits_without_news[~complete_hits_without_news['is_bot']==True]
print('complete hits after removing bot authors: ', complete_hits.shape)

hits_per_author = complete_hits.groupby(['post_author'])["post_id"].count().reset_index(name="count")
hits_per_author = hits_per_author.sort_values(by='count', ascending=False)
hits_per_author = hits_per_author[hits_per_author['post_author'] != '[deleted]']

chunk_size = 10000
initial_value = 0
end_value = chunk_size
count = 0
while (initial_value < (hits_per_author.shape)[0]):
    author_status = []
    author_data = hits_per_author[initial_value:end_value]
    for author in author_data['post_author'].tolist():
        url = 'https://www.reddit.com/user/'+ author +'.json'
        try:
            resp = requests.get(url,headers = {'User-agent': 'your bot 0.1'})
            author_status.append(resp.status_code)
        except Exception as e:
            print(e)
            print(author)
            author_status.append(None)
    author_data['author_status'] = author_status
    author_data.to_csv('./results/author_status/'+str(count)+'.csv')
    count = count + 1
    initial_value = end_value
    end_value = end_value + chunk_size
    print('count', count)

# author_status = []
# count = 0
# for author in hits_per_author['post_author'].tolist():
    
#     url = 'https://www.reddit.com/user/'+ author +'.json'
#     try:
#         resp = requests.get(url,headers = {'User-agent': 'your bot 0.1'})
#         author_status.append(resp.status_code)
#         count = count + 1
#         print('count', count)
#     except Exception as e:
#         print(e)
#         print(author)
#         author_status.append(None)
        
# hits_per_author['author_status'] = author_status
# hits_per_author.to_csv('./results/author_status.csv')
    