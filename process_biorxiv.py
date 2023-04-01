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
from parse_ocr import detect_ocr
import fitz
import xmltodict
import boto3
import zipfile



parser = argparse.ArgumentParser()
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=1)


def get_text(doc):
    text_list = []
 
    doc = fitz.open(stream=doc)
    for i in range(doc.page_count):
        page = doc.load_page(i)
        text = page.get_text()
        if text != '':
            text_list.append(text)
        else:
            text = detect_ocr(page.get_pixmap())
            text_list.append(text)
    if len(text_list) > 0:
        return '\n'.join(text_list)

def get_data(f, output_dir, client):
    try:
        obj = client.get_object(Bucket='biorxiv-src-monthly', Key=f['Key'],RequestPayer='requester')

        with zipfile.ZipFile(io.BytesIO(obj['Body'].read())) as z:
            for name in z.namelist():
                f_name = name.split('/')[-1].split('.')[0]
                if name.endswith('.xml'):
                    with open(f'{output_dir}/{f_name}.json', 'w') as f:
                        json.dump(xmltodict.parse(z.read(name)), f)
                if name.endswith('.pdf') and 'file' not in name:
                    doc = z.read(name)
                    # text = get_text(doc)
                    with open(f'{output_dir}/{f_name}.pdf', 'wb') as f:
                        f.write(doc)
    except:
        return


def process_part(files, output_dir, st, client):
    i = 0
    data = {
        'TEXT':[],
        'SOURCE':[]
    }
    for f in tqdm(files):
        try:
            get_data(f, output_dir, client)
                        
        except:
            # print(key)
            continue

def process_files(fs, output_dir, st):
    data = {
        'TEXT': [],
        'YEAR': [],
        'TITLE': [],
        'DOI': [],
        'AUTHORS': [],
        'LICENSE': []
    }

    i = 1

    for f in tqdm(fs):
        json_path = f.replace('.txt', '.json')
        try:
            with open(json_path) as fh:
                d = json.load(fh)

            title = d['article']['front']['article-meta']['title-group']['article-title']
            if type(title) == dict:
                title = title['#text']
            try:
                authors = [' '.join([a['name']['surname'], a['name']['given-names']]) for a in d['article']['front']['article-meta']['contrib-group']['contrib']]
            except:
                authors = None
            try:
                year = d['article']['front']['article-meta']['pub-date']['year']
            except:
                year = None
            doi = d['article']['front']['article-meta']['article-id']['#text']
            lic = d['article']['front']['article-meta']['permissions']['license'].get('@license-type', None)
            with open(f, 'r') as fh:
                text = fh.read()
            data['TEXT'].append(text)
            data['YEAR'].append(year)
            data['TITLE'].append(title)
            data['DOI'].append(doi)
            data['AUTHORS'].append(authors)
            data['LICENSE'].append(lic)

            if i > 0 and i % 100 == 0:
                df = pd.DataFrame(data)
                df.to_parquet(f'{output_dir}/{i}_{st}.parquet', index=False)
                data = {
                    'TEXT': [],
                    'YEAR': [],
                    'TITLE': [],
                    'DOI': [],
                    'AUTHORS': [],
                    'LICENSE': []
                }
            i += 1
        except Exception as err:
            print(err)
            continue
 


if __name__=='__main__':


    output_dir = f'BIORXIV_raw'
    os.makedirs(output_dir, exist_ok=True)
    client = boto3.client('s3')
    
    paginator = client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket='biorxiv-src-monthly',RequestPayer='requester', Prefix='Current_Content/')


    s = time.time()
    

    for i, page in enumerate(pages):
       
        files = list(page['Contents'])
        N = len(files)
        print(N)
        processes = []
        num_process = 64
        files = page['Contents']
        with ThreadPool(num_process) as p:
            p.starmap(get_data, [(f, output_dir, client) for f in files])

    # convert to parquets
    output_dir = 'biorxiv_parquets'
    os.makedirs(output_dir, exist_ok=True)

    files = glob.glob('BIORXIV/**/*.txt', recursive=True)
    print(len(files))
    N = len(files)
    num_process = 20
    processes = []
    rngs = [(i*int(N/num_process), (i+1)*int(N/num_process))
            for i in range(num_process)]
    print(rngs)
    for rng in rngs:
        start, end = rng
        p = mp.Process(target=process_files, args=[
                    files[start:end], output_dir, start])
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    e = time.time()
    print(e-s)
            