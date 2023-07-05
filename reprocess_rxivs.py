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
import multiprocessing.pool
import pandas as pd
import fitz
import boto3
import layoutparser as lp
import zipfile
import cv2
from huggingface_hub import hf_hub_download, HfApi
import numpy as np
from resiliparse.parse.html import HTMLTree

ckpt_path = hf_hub_download('marianna13/chemical_layout_parser', 'model_final.pth')

model = lp.models.Detectron2LayoutModel('lp://PubLayNet/mask_rcnn_R_50_FPN_3x/config',
                                        ckpt_path,
                                        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
                                        label_map={ 0: 'figure',  1: 'formula', 2: 'molecule', 3: "references", 4:'table', 5: 'text'})




parser = argparse.ArgumentParser()
parser.add_argument('--repo', type=str)
parser.add_argument('--token', type=str)
parser.add_argument('--num_process', type=int, default=32)
parser.add_argument('--output_dir', type=str)
parser.add_argument('--img_dir', type=str)
parser.add_argument('--dataset_name', type=str, default='biorxiv')

def get_block(coords, p, h, w):
    style = {s.split(':')[0]:float(s.split(':')[1].replace('pt', '')) for s in p['style'].split(';')}
    x1, y1, x2, y2 = coords
    return y1 <= style['top'] and y2 >= style['top']


def get_replace_dict(tree):
  replace_dict = {}
  for ele in tree.body.get_elements_by_tag_name('p'):
    text = ''
    if len(ele.text.replace(' ', '').replace('\n', '')) == 0:
      continue
    for s in ele.get_elements_by_tag_name('*'):

      style = s.getattr('style')
      font_size = None
      if style:

        style = {st.split(':')[0]:st.split(':')[1]  for st in style.split(';')}

        font_size = float(style['font-size'].replace('pt', ''))

        if (font_size and font_size == 14) or s.parent.tag == 'b':
          text += f'<b> {s.text} </b>'
        elif font_size and font_size < 7 and s.parent.tag != 'sup':
          text += f'<sub>{s.text}</sub>'
        elif s.tag == 'span' and s.parent.tag != 'sup':
          text += s.text
        elif s.tag == 'sup':
          text += f'<sup>{s.text}</sup>'

    replace_dict[ele.text] = text

  return replace_dict

def contains(coords, b_coords):
  x1, y1, x2, y2 = coords
  x1_, y1_, x2_, y2_ = b_coords
  return y1_ >= y1 and y2_ <= y2 and x1_ >= x1 and x2_ <= x2

def process_val(val):
  text = ''
  val_dict = {}
  texts = 0
  titles = 0
  for v in val:


    if type(v) == str :
      v = v.strip()
      if v.startswith('<b>') and v.endswith('</b>'):
        val_dict[f'title_{titles}'] = v.replace('<b>', '').replace('</b>', '')
        titles += 1
      else:
        val_dict[f'text_{texts}'] = v
        texts += 1
    elif type(v) == dict:

      for k in v.keys():
        val_dict[k] = v[k]

  return val_dict


  def get_blocks_dict(page, page_no, img_dir):
    html = HTMLTree.parse(page.get_text('html'))
    h, w = page.rect.width, page.rect.height
    # print(str(html))
    replace_dict = get_replace_dict(html)
    blocks = page.get_text('blocks')

    # blocks.sort(key = lambda b:b[0])

    imgs, tables = 0, 0
    num_nl = None
    table = []
    keys = []
    pix = page.get_pixmap()
    pix = cv2.imdecode(np.frombuffer(pix.tobytes('jpg'), np.uint8), -1)
    layout = model.detect(pix)
    figure_blocks = [list(b.coordinates)+[b.type, 1] for b in layout if b.type in ['figure', 'formula', 'molecule', 'table']]
    # print(figure_blocks)
    blocks.extend(figure_blocks)
    blocks.sort(key = lambda b: (b[0], b[1]))
    blocks_dict = {f'block_{i}': [] for i in range(len(blocks))}
    for i, b in enumerate(blocks):
      if b[-1] == 1:
        x1, y1, x2, y2 = map(int, b[:4])
        im_type = b[-2]
        # if im_type not in ['figure', 'formula', 'molecule', 'table']:
        #   im_type = 'figure'
        img = pix[y1:y2, x1:x2]
        try:
          cv2.imwrite(f'{img_dir}/{page_no}_{imgs}.png', img)
          block_text = {
              f'image_{imgs}': f'{page_no}_{imgs}.png',
              'coords': [x1, y1, x2, y2],
              'fig_type': im_type
          }
          imgs += 1
        except Exception as err:
          print(err)
          continue

      else:
        block_text = b[4]
        for t in block_text.split('\n'):
          try:
            block_text = block_text.replace(t, replace_dict[t])
          except Exception as err:
            # print(err)
            continue

      blocks_dict[f'block_{i}'].append(block_text)
      keys.append(f'block_{i}')


    # blocks_dict = {k: process_val(blocks_dict[k]) for k in blocks_dict.keys()}
    blocks_dict = sorted(blocks_dict.items(), key=lambda pair: keys.index(pair[0]))
    return blocks_dict


def get_blocks_dict(page, page_no, img_dir):
    html = HTMLTree.parse(page.get_text('html'))
    h, w = page.rect.width, page.rect.height
    # print(str(html))
    replace_dict = get_replace_dict(html)
    blocks = page.get_text('blocks')

    # blocks.sort(key = lambda b:b[0])

    imgs, tables = 0, 0
    num_nl = None
    table = []
    keys = []
    pix = page.get_pixmap()
    pix = cv2.imdecode(np.frombuffer(pix.tobytes('jpg'), np.uint8), -1)
    layout = model.detect(pix)
    figure_blocks = [list(b.coordinates)+[b.type, 1] for b in layout if b.type in ['figure', 'formula', 'molecule', 'table']]
    # print(figure_blocks)
    blocks.extend(figure_blocks)
    blocks.sort(key = lambda b: (b[0], b[1]))
    blocks_dict = {f'block_{i}': [] for i in range(len(blocks))}
    for i, b in enumerate(blocks):
      if b[-1] == 1:
        x1, y1, x2, y2 = map(int, b[:4])
        im_type = b[-2]
        # if im_type not in ['figure', 'formula', 'molecule', 'table']:
        #   im_type = 'figure'
        img = pix[y1:y2, x1:x2]
        try:
          cv2.imwrite(f'{img_dir}/{page_no}_{imgs}.png', img)
          block_text = {
              f'image_{imgs}': f'{page_no}_{imgs}.png',
              'coords': [x1, y1, x2, y2],
              'fig_type': im_type
          }
          imgs += 1
        except Exception as err:
          print(err)
          continue

      else:
        block_text = b[4]
        for t in block_text.split('\n'):
          try:
            block_text = block_text.replace(t, replace_dict[t])
          except Exception as err:
            # print(err)
            continue

      blocks_dict[f'block_{i}'].append(block_text)
      keys.append(f'block_{i}')



    # blocks_dict = {k: process_val(blocks_dict[k]) for k in blocks_dict.keys()}
    blocks_dict = sorted(blocks_dict.items(), key=lambda pair: keys.index(pair[0]))
    return blocks_dict

def get_data(f, output_dir, img_dir, client, bucket):

    try:
        obj = client.get_object(Bucket=bucket, Key=f['Key'],RequestPayer='requester')


        with zipfile.ZipFile(io.BytesIO(obj['Body'].read())) as z:
            for name in z.namelist():
                doc_pages = {}
                if name.endswith('.pdf') and 'file' not in name:
                    doc = fitz.open(stream=io.BytesIO(z.read(name)))
                    fig_dir = f'{img_dir}/'+name.split('/')[-1]
                    os.makedirs(fig_dir, exist_ok=True)
                    for i, page in enumerate(doc.pages()):
                        pix = page.get_pixmap()
                        pix = cv2.imdecode(np.frombuffer(pix.tobytes('jpg'), np.uint8), -1)
                        
                        try:
                            blocks_dict = get_blocks_dict(page, i, fig_dir)
                            doc_pages[f'page_{i}'] = blocks_dict
                        except Exception as err:
                            print(err)
                            continue
                    j_name = f'{output_dir}/' + name.split('/')[-1].replace('.pdf', '.json')
                    with open(j_name, 'w') as f:
                        json.dump(doc_pages, f)

             
    except Exception as er:
        print(er)
        return




if __name__=='__main__':

    args = parser.parse_args()

    api = HfApi()
    REPO = args.repo
    TOKEN = args.token
    dataset = args.dataset_name


    output_dir = args.output_dir
    img_dir = args.img_dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    client = boto3.client('s3')
    
    paginator = client.get_paginator('list_objects_v2')
    bucket = f'{dataset}-src-monthly'
    pages = paginator.paginate(Bucket=bucket, RequestPayer='requester', Prefix='Current_Content/')

    for i, page in enumerate(pages):
       
        files = list(page['Contents'])[:1]
        N = len(files)
        print(N)
        processes = []
        num_process = args.num_process
        files = [f for f in page['Contents'] if f['Key'].endswith('.meca')]
        with ThreadPool(num_process) as p:
            p.starmap(get_data, [(f, output_dir, img_dir, client, bucket) for f in files])
