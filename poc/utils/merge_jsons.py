import json
import glob

# result = []
# for f in glob.glob("/INET/state-trolls/work/state-trolls/reddit_dataset/comments/RC_2016-01.bz2.decompressed-*.json"):
#     with open(f, "r") as infile:
#         try:
#             result.extend(json.load(infile))
#         except ValueError:
#             print(f)

# with open("/INET/state-trolls/work/state-trolls/reddit_dataset/comments/RC_2016-01.scores.json", "w") as outfile:
#     json.dump(result, outfile)

with open('/INET/state-trolls/work/state-trolls/reddit_dataset/comments/RC_2016-01.scores-merged.json', 'w') as outfile:
    for fname in glob.glob("/INET/state-trolls/work/state-trolls/reddit_dataset/comments/RC_2016-01.bz2.decompressed-*.json"):
        print(fname)
        with open(fname) as infile:
            for line in infile:
                outfile.write(line)
