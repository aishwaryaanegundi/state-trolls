import numpy as np 
import pandas as pd
import string
import json

data = []
try:
    with open('/INET/state-trolls/work/state-trolls/reddit_dataset/comments/RC_2016-01.scores-merged.json', 'r') as content_file:
        content = content_file.read()
        print('file content read')
        jsons = content.replace('}','}#')
        print('replacement done')
        jsons = jsons.split('#')
        print('processed')
    for element in jsons:
        d = eval(element)
        print(d)
        data.append(d)
except:
    print('exception occured')
print(len(data))
# print(data)
with open('/INET/state-trolls/work/state-trolls/reddit_dataset/comments/scores/RC_2016-01.scores.json', 'w') as fout:
    json.dump(data, fout)
