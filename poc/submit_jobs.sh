#!/bin/bash

# list of files for which encodings has to be obtained
file_list="RC_2016-12.bz2.decompressed RC_2017-01.bz2.decompressed RC_2017-02.bz2.decompressed RC_2017-03.bz2.decompressed RC_2017-04.bz2.decompressed RC_2017-05.bz2.decompressed RC_2017-06.bz2.decompressed"

for f in ${file_list}; do
    echo 'The input file to be encoded is:' ${f} 
    python setup.py
    status='0'
#     read -r status<'job_status.txt'
#     status = $(head -n 1 'job_status.txt')
    while [ ${status} == '0' ];
    do
        sbatch bert_server.sh
        sbatch bert_client.sh ${f}
        sleep 26h
        read -r status<'job_status.txt'
    done
done;
