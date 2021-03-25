from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import misc.utils as utils
from collections import OrderedDict
import torch
import math
import sys
sys.path.append("cider")
from pyciderevalcap.ciderD.ciderD import CiderD
from pyciderevalcap.cider.cider import Cider
sys.path.append("coco-caption")
from pycocoevalcap.bleu.bleu import Bleu

CiderD_scorer = None
Cider_scorer = None
Bleu_scorer = None
#CiderD_scorer = CiderD(df='corpus')

def init_scorer(cached_tokens):
    global CiderD_scorer
    CiderD_scorer = CiderD_scorer or CiderD(df=cached_tokens)
    global Cider_scorer
    Cider_scorer = Cider_scorer or Cider(df=cached_tokens)
    global Bleu_scorer
    Bleu_scorer = Bleu_scorer or Bleu(4)

def array_to_str(arr):
    out = ''
    for i in range(len(arr)):
        out += str(arr[i]) + ' '
        if arr[i] == 0:
            break
    return out.strip()

def get_self_critical_reward(greedy_res, data_gts, gen_result, opt):
    batch_size = gen_result.size(0)# batch_size = sample_size * seq_per_img
    seq_per_img = batch_size // len(data_gts)

    res = OrderedDict()
    
    gen_result = gen_result.data.cpu().numpy()
    greedy_res = greedy_res.data.cpu().numpy()
    for i in range(batch_size):
        res[i] = [array_to_str(gen_result[i])]
    for i in range(batch_size):
        res[batch_size + i] = [array_to_str(greedy_res[i])]

    gts = OrderedDict()
    for i in range(len(data_gts)):
        gts[i] = [array_to_str(data_gts[i][j]) for j in range(len(data_gts[i]))]

    res_ = [{'image_id':i, 'caption': res[i]} for i in range(2 * batch_size)]
    res__ = {i: res[i] for i in range(2 * batch_size)}
    gts = {i: gts[i % batch_size // seq_per_img] for i in range(2 * batch_size)}
    if opt.cider_reward_weight > 0:
        _, cider_scores = CiderD_scorer.compute_score(gts, res_)
        #print('Cider scores:', _)
    else:
        cider_scores = 0
    if opt.bleu_reward_weight > 0:
        _, bleu_scores = Bleu_scorer.compute_score(gts, res__)
        bleu_scores = np.array(bleu_scores[3])
        print('Bleu scores:', _[3])
    else:
        bleu_scores = 0
    scores = opt.cider_reward_weight * cider_scores + opt.bleu_reward_weight * bleu_scores

    scores = scores[:batch_size] - scores[batch_size:]

    rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)

    return rewards

def cal_cider(data_gts, gen_result):
    batch_size = 1# batch_size = sample_size * seq_per_img
    res = OrderedDict()
    for i in range(batch_size):
        res[i] = [gen_result]
    gts = OrderedDict()
    for i in range(batch_size):
        gts[i] = data_gts

    res_ = [{'image_id':i, 'caption': res[i]} for i in range(batch_size)]
    _, cider_scores = CiderD_scorer.compute_score(gts, res_)
    #print("[gd:",data_gts,"] [generation:",gen_result,"]-->",_)
    return _ 

def get_scores_separate(data_gts, gen_result):
    allscore ={}
    batch_size = len(gen_result)# batch_size = sample_size * seq_per_img
    seq_per_img = batch_size//len(data_gts)

    res = OrderedDict()
    
    #gen_result = gen_result.data.cpu().numpy()
    for i in range(batch_size):
        res[i] = [gen_result[i]]
    gts = OrderedDict()
    for i in range(len(data_gts)):
        gts[i] = data_gts[i]
        #gts[i] = [data_gts[i][j] for j in range(len(data_gts[i]))]
    res_ = [{'image_id':i, 'caption': res[i]} for i in range(batch_size)]
    res__ = {i: res[i] for i in range(batch_size)}
    gts = {i: gts[i // seq_per_img] for i in range(batch_size)}
    cider_score, cider_scores = Cider_scorer.compute_score(gts, res_)
    #print('Cider score:', cider_score)
    allscore['Cider'] = cider_score
    bleu_score, bleu_scores = Bleu_scorer.compute_score(gts, res__)
    for index, b in enumerate(bleu_score):
        allscore["Bleu"+str(index+1)]=b
    bleu_scores = np.array(bleu_scores[3])
    #print('Bleu scores:', _[3])
    return allscore 

