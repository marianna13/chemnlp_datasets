
import io
import json
from tqdm import tqdm
import os
import time
import glob
import multiprocessing as mp
import json
import argparse
from multiprocessing.pool import ThreadPool
import pandas as pd
import fitz
import xmltodict
import boto3
import cv2
import zipfile
import numpy as np




def get_data(f, output_dir, client):
    
       
    try:
        obj = client.get_object(Bucket='medrxiv-src-monthly', Key=f['Key'],RequestPayer='requester')
        with zipfile.ZipFile(io.BytesIO(obj['Body'].read())) as z:
            for name in z.namelist():
                f_name = name.split('/')[-1].split('.')[0]
                
                if name.endswith('.xml'):
                    d = xmltodict.parse(z.read(name))
                    new_d = {}
                    new_d['title'] = d['article']['front']['article-meta']['title-group']['article-title']
                    new_d['contrib'] = d['article']['front']['article-meta']['contrib-group']['contrib']
                    new_d['abstract'] = d['article']['front']['article-meta']['abstract']
                    for sec in d['article']['body']['sec']:
                        new_d[sec['title']] = {k: sec[k] for k in sec.keys() if k != 'title'}
                    new_d['References'] = d['article']['back']['ref-list']['ref']
                    with open(f'{output_dir}/{f_name}.json', 'w') as f:
                        json.dump(d, f)

                  
    except Exception as err:
        print(err)
        print(d)
        return


def process_part(files, output_dir, st, client):
    i = 0
    data = {
        'TEXT':[],
        'SOURCE':[]
    }
    for f in tqdm(files):
            # print(f)
        try:
            get_data(f, output_dir, client)
                        
        except:
            # print(key)
            continue


if __name__=='__main__':


    output_dir = f'biorxiv_xml'
    os.makedirs(output_dir, exist_ok=True)
    client = boto3.client('s3')
    
    paginator = client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket='biorxiv-src-monthly',RequestPayer='requester', Prefix='Current_Content/')

    s = time.time()
    

    for i, page in enumerate(pages):
       
        files = list(page['Contents'])
        N = len(files)
        print(N)
        files = page['Contents']
        files = [f for f in files if f['Key'].endswith('meca')]
        process_part(files, output_dir, 0, client)
  
        


            
