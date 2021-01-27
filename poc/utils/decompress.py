import sys
import os
import bz2
from bz2 import decompress

path = "/INET/state-trolls/nobackup/reddit-data/"
new_path = "/INET/state-trolls/work/state-trolls/reddit_dataset/"
i = 0 
for(dirpath,dirnames,files)in os.walk(path):
    dirs = os.listdir(path)
#     print(dirs[0])
#    for file in files:
#        filepath = os.path.join(dirpath,filename)
#        newfile = bz2.decompress(file)
#        newfilepath = os.path.join(dirpath,newfile)
        
    for filename in files:
        filepath = os.path.join(dirpath, filename)
        newfilepath = os.path.join(new_path, dirs[i], filename + '.decompressed')
        print(newfilepath)
        with open(newfilepath, 'wb') as new_file, bz2.BZ2File(filepath, 'rb') as file:
            for data in iter(lambda : file.read(100 * 1024), b''):
                new_file.write(data)
    i = i + 1