"""
Preprocess a raw json dataset into hdf5/json files for use in data_loader.lua

Input: json file that has the form
[{ file_path: 'path/img.jpg', captions: ['a caption', ...] }, ...]
example element in this list would look like
{'captions': [u'A man with a red helmet on a small moped on a dirt road. ', u'Man riding a motor bike on a dirt road on the countryside.', u'A man riding on the back of a motorcycle.', u'A dirt path with a young person on a motor bike rests to the foreground of a verdant area with a bridge and a background of cloud-wreathed mountains. ', u'A man in a red shirt and a red hat is on a motorcycle on a hill side.'], 'file_path': u'val2014/COCO_val2014_000000391895.jpg', 'id': 391895}

This script reads this json, does some basic preprocessing on the captions
(e.g. lowercase, etc.), creates a special UNK token, and encodes everything to arrays

Output: a json file and an hdf5 file
The hdf5 file contains several fields:
/images is (N,3,256,256) uint8 array of raw image data in RGB format
/labels is (M,max_length) uint32 array of encoded labels, zero padded
/label_start_ix and /label_end_ix are (N,) uint32 arrays of pointers to the 
  first and last indices (in range 1..M) of labels for each image
/label_length stores the length of the sequence for each of the M sequences

The json file has a dict that contains:
- an 'ix_to_word' field storing the vocab in form {ix:'word'}, where ix is 1-indexed
- an 'images' field that is a list holding auxiliary information for each image, 
  such as in particular the 'split' it was assigned to.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import argparse
from random import shuffle, seed
import string
# non-standard dependencies:
import h5py
import numpy as np
import torch
import torchvision.models as models
import skimage.io
from PIL import Image
from torch import nn
import string
punc=string.punctuation.replace("'", "")
#info = json.load(open("data/personcap.json"))
info = {}
added= False
import re
RETOK = re.compile(r'\w+|[^\w\s]|\n', re.UNICODE)

def build_vocab(imgs, params):
  count_thr = params['word_count_threshold']
  rm_punc=params['rm_punc']
  # count up the number of words
  counts = {}
  split_vocab ={"train":[],"test":[],"val":[]}
  avg_cap_len ={"train":[],"test":[],"val":[]}
  for img in imgs:
      for sent in img['sentences']:
          if isinstance(sent,dict) and 'tokens' in sent:
              tokens=sent['tokens']
          elif isinstance(sent,str):
              sent=sent.lower()
              if rm_punc>0:
                  sent=sent.translate(str.maketrans('', '', punc))
              tokens= RETOK.findall(sent)
          for w in tokens:
              counts[w] = counts.get(w, 0) + 1
              split_vocab[img['split']].append(w)
          avg_cap_len[img['split']].append(len(tokens))
  for key,val  in split_vocab.items():
      print(key,": ",len(set(val)))
      print('avg_len',np.mean(np.array(avg_cap_len[key])))

  cw = sorted([(count,w) for w,count in counts.items()], reverse=True)
  print('top words and their counts:')
  print('\n'.join(map(str,cw[:20])))

  # print some stats
  total_words = sum(counts.values())
  print('total words:', total_words)
  bad_words = [w for w,n in counts.items() if n <= count_thr] 
  if 'ix_to_word' in info and added:
      vocab = list(info['ix_to_word'].values())
#      for w,n in counts.items():
#          if n > count_thr and w not in vocab:
#              vocab.append(w)
  else:
      vocab = [w for w,n in counts.items() if n > count_thr]
  bad_count = sum(counts[w] for w in bad_words)
  print('number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words)*100.0/len(counts)))
  print('number of words without UNK in vocab would be %d' % (len(vocab), ))
  print('number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count*100.0/total_words))
  # lets look at the distribution of lengths as well
  sent_lengths = {}
  for img in imgs:
      for sent in img['sentences']:
          if isinstance(sent,dict) and 'tokens' in sent:
              txt=sent['tokens']
          elif isinstance(sent,str):
              sent=sent.lower()
              if rm_punc>0:
                  sent=sent.translate(str.maketrans('', '', punc))
              txt= RETOK.findall(sent)
              if('[' in txt and len(txt)==3):
                  #print(sent)
                  continue
              nw = len(txt)
              sent_lengths[nw] = sent_lengths.get(nw, 0) + 1
  max_len = max(sent_lengths.keys())
  print('max length sentence in raw data: ', max_len)
  print('sentence length distribution (count, number of words):')
  sum_len = sum(sent_lengths.values())
  for i in range(max_len+1):
      print('%2d: %10d   %f%%' % (i, sent_lengths.get(i,0), sent_lengths.get(i,0)*100.0/sum_len))

  # lets now produce the final annotations
  if bad_count > 0 and not added:
      # additional special UNK token we will use below to map infrequent words to
    print('inserting the special UNK token')
    vocab.append('UNK')
  
  
  for img in imgs:
      img['final_captions'] = []
      for sent in img['sentences']:
          if isinstance(sent,dict) and 'tokens' in sent:
              txt=sent['tokens']
          elif isinstance(sent,str):
              sent=sent.lower()
              if rm_punc>0:
                  sent=sent.translate(str.maketrans('', '', punc))
              txt= RETOK.findall(sent)
              if('[' in txt and len(txt)==3):
                  #print(sent)
                  continue
          caption = [w if counts.get(w,0) > count_thr else 'UNK' for w in txt]
          img['final_captions'].append(caption)
  return vocab

def encode_captions(imgs, params, wtoi,cap):
    """ 
  encode all captions into one large array, which will be 1-indexed.
  also produces label_start_ix and label_end_ix which store 1-indexed 
  and inclusive (Lua-style) pointers to the first and last caption for
  each image in the dataset.
  """
    max_length = params['max_length']
    N = len(imgs)
    M = sum(len(img['final_captions']) for img in imgs) # total number of captions
    rm_punc=params['rm_punc']
    label_arrays = []
    label_start_ix = np.zeros(N, dtype='uint32') # note: these will be one-indexed
    label_end_ix = np.zeros(N, dtype='uint32')
    label_length = np.zeros(M, dtype='uint32')
    cap_arrays = []
    cap_start_ix = np.zeros(N, dtype='uint32') # note: these will be one-indexed
    cap_end_ix = np.zeros(N, dtype='uint32')
    cap_length = np.zeros(M*5, dtype='uint32')
    label_counter = 0
    caption_counter = 0
    counter = 1
    nzeros =0
    invalid_hash = []
    for i,img in enumerate(imgs):
        img_hash=img['image_hash']
        if img_hash in cap:
            fivecaps=cap[img_hash]['captions'][:5]
        else:
            print(img_hash)
        n = len(img['final_captions'])
        if n==0:
            nzeros = nzeros +1
            invalid_hash.append(img_hash)
            continue        # no final caption means we remove some invaid sentences
        assert n > 0, 'error: some image has no captions'
        Li = np.zeros((n, max_length), dtype='uint32')
        Ci = np.zeros((5, max_length), dtype='uint32')
        for j,s in enumerate(fivecaps):
            s=s.lower()
            if rm_punc>0:
                s=s.translate(str.maketrans('', '', punc))
            txt= RETOK.findall(s)
            cap_length[caption_counter] = min(max_length, len(txt)) # record the length of this sequence
            caption_counter += 1
            for k,w in enumerate(txt):
                if k < max_length:
                    Ci[j,k] = wtoi.get(w,wtoi['UNK'])
        for j,s in enumerate(img['final_captions']):
            label_length[label_counter] = min(max_length, len(s)) # record the length of this sequence
            label_counter += 1
            for k,w in enumerate(s):
                if k < max_length:
                    Li[j,k] = wtoi.get(w,wtoi['UNK'])
    # note: word indices are 1-indexed, and captions are padded with zeros
        label_arrays.append(Li)
        label_start_ix[i] = counter
        label_end_ix[i] = counter + n - 1
        
        cap_arrays.append(Ci)
        counter += n
    C = np.array(cap_arrays)
    L = np.concatenate(label_arrays, axis=0) # put all the labels together
    print(nzeros)
    assert L.shape[0] == M, 'lengths don\'t match? that\'s weird'
    assert np.all(label_length > 0), 'error: some caption had no words?'

    print('encoded captions to array of size ', L.shape)
    return L, label_start_ix, label_end_ix, label_length,C,cap_length,invalid_hash

def load_personalities(personality_path):
  perss = []
  with open(personality_path) as f:
      for line in f:
          if 'Trait' not in line:
              perss.append(line[0:-1])
  return perss
#def build_personality_encoder(num_personalities,dropout,hidden_dim):
#    personality_layers = [
#            nn.BatchNorm1d(num_personalities),
#            nn.Dropout(p=dropout),
#            nn.Linear(num_personalities, hidden_dim),
#        ]
#    personality_encoder = nn.Sequential(*personality_layers)
#    return personality_encoder

def onehot_personalities(imgs,params,perss,ptoi):
    N = len(imgs)
    personality_arrays = []
    num_personalities = len(perss) + 1
    res = np.zeros((N, num_personalities),dtype=np.float64)
    counter = 1
    rm_punc=params['rm_punc']
    for i,img in enumerate(imgs):
        n = len(img['personality'])
        assert n > 0, 'error: some image has no personality'
        p =img['personality']
        if p in ptoi:
            res[i,ptoi[p]]=1
        elif p=="Earnest":
            res[i,ptoi["Earnest (Enthusiastic)"]]=1
        else:
            print(p)
    return res

def main(params):

    imgs = json.load(open(params['input_json'], 'r'))
    imgs = imgs['images']
    print(len(imgs))
    imgs=[img for img in imgs if not img['image_hash'].startswith('ac8')]
    imgs = [img for img in imgs if '[DISCONNECT]' not in img['sentences'][0]  and '[TIMEOUT]' not in img['sentences'][0] and '[RETURNED]' not in img['sentences'][0]]
    print(len(imgs))
    seed(123) # make reproducible
    # create the vocab
    vocab = build_vocab(imgs, params) 
    itow = {i+1:w for i,w in enumerate(vocab)} # a 1-indexed vocab translation table
    wtoi = {w:i+1 for i,w in enumerate(vocab)} # inverse table
    # load captions from caption_path 
    cap=json.load(open(params['caption_path'],'r'))
    print("cap length,",len(cap))
    # encode captions in large arrays, ready to ship to hdf5 file
    L, label_start_ix, label_end_ix, label_length,C,cap_length,invalid_hash = encode_captions(imgs, params, wtoi,cap)
    #load personality and turn it into one-hot
    perss=load_personalities(params["personality_path"])
    perss.append("Crude")
    itop = {i+1:p for i,p in enumerate(perss)} # a 1-indexed perss translation table
    ptoi = {p:i+1 for i,p in enumerate(perss)} # inverse table
    onehot_perss = onehot_personalities(imgs,params,perss,ptoi)
    # create output h5 file
    N = len(imgs)
    f_lb = h5py.File(params['output_h5']+'_label.h5', "w")
    f_lb.create_dataset("labels", dtype='uint32', data=L)
    f_lb.create_dataset("label_start_ix", dtype='uint32', data=label_start_ix)
    f_lb.create_dataset("label_end_ix", dtype='uint32', data=label_end_ix)
    f_lb.create_dataset("label_length", dtype='uint32', data=label_length)
    f_lb.close()
    # create to save personality-one-hot
    f_lp = h5py.File(params['perssOnehot_h5']+'.h5', "w")
    f_lp.create_dataset("perss_onehot", dtype='uint32', data=onehot_perss)
    f_lp.close()
    #create output caption h5 file
    f_lc = h5py.File(params['densecap_h5']+'.h5', "w")
    f_lc.create_dataset("dense_cap", dtype='uint32', data=C)
    f_lc.create_dataset("cap_length", dtype='uint32', data=cap_length)
    f_lc.close()
    # create output json file
    out = {}
    out['ix_to_word'] = itow # encode the (1-indexed) vocab
    out['images'] = []
    out['pix_to_personality']=itop
    for i,img in enumerate(imgs):
        if img['image_hash'] in invalid_hash:
            print(img['image_hash'])
            continue
        jimg = {}
        jimg['split'] = img['split']
        if 'filename' in img: jimg['file_path'] = os.path.join(img.get('filepath', ''), img['filename']) # copy it over, might need
        if 'cocoid' in img:
            jimg['id'] = img['cocoid'] # copy over & mantain an id, if present (e.g. coco ids, useful)
        elif 'imgid' in img:
            jimg['id'] = img['imgid']

        if params['images_root'] != '':
            with Image.open(os.path.join(params['images_root'], img['filepath'], img['filename'])) as _img:
                jimg['width'], jimg['height'] = _img.size
        if 'personality' in img:
            jimg['personality']=img['personality']
        if 'candidates' in img:
            jimg['candidates']=img['candidates']
        jimg['sentence']=img['final_captions']
        #print(jimg['sentence'])
        out['images'].append(jimg)

    json.dump(out, open(params['output_json'], 'w'))
    print('wrote ', params['output_json'])

if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  # input json
  parser.add_argument('--input_json',default=' ParlAI/data/personality_captions/dataset_person.json ', required=True, help='input json file to process into hdf5')
  parser.add_argument('--output_json', default='data/personcap.json', help='output json file')
  parser.add_argument('--perssOnehot_h5', default='data/person_onehot', help='output h5 file for personality one hot')  
  parser.add_argument('--densecap_h5', default='data/densecap', help='output h5 file for personality one hot')
  parser.add_argument('--output_h5', default='data/personcap', help='output h5 file')
  parser.add_argument('--images_root', default='', help='root location in which images are stored, to be prepended to file_path in input json')
  parser.add_argument('--personality_path', default='ParlAI/data/personality_captions/personalities.txt', help='root location in which personalities are stored')
  parser.add_argument('--caption_path', default='densecap/img_caption.json', help='root location in which personalities are stored')
  # options
  parser.add_argument('--max_length', default=16, type=int, help='max length of a caption, in number of words. captions longer than this get clipped.')
  parser.add_argument('--word_count_threshold', default=5, type=int, help='only words that occur more than this number of times will be put in vocab')
  parser.add_argument('--rm_punc', default=0, type=int, help='remove punctuation')
  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  print('parsed input parameters:')
  print(json.dumps(params, indent = 2))
  main(params)
