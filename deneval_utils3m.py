from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

import numpy as np
import json
from json import encoder
import random
import string
import time
import os
import sys
import misc.utils as utils
from misc.rewards import init_scorer, cal_cider,get_scores_separate
import pandas as pd
import numpy as np
import random
bad_endings = ['a','an','the','in','for','at','of','with','before','after','on','upon','near','to','is','are','am']
bad_endings += ['the']

def count_bad(sen):
    sen = sen.split(' ')
    if sen[-1] in bad_endings:
        return 1
    else:
        return 0

def language_eval(dataset, preds, model_id, split):
    import sys
    sys.path.append("coco-caption")
    if 'coco' in dataset:
        annFile = 'coco-caption/annotations/captions_val2014.json'
    elif 'flickr30k' in dataset or 'f30k' in dataset:
        annFile = 'coco-caption/f30k_captions4eval.json'
    elif 'person' in dataset:
        annFile='coco-caption/person_captions4eval.json'
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap
    # encoder.FLOAT_REPR = lambda o: format(o, '.3f')

    if not os.path.isdir('eval_results'):
        os.mkdir('eval_results')
    cache_path = os.path.join('eval_results/', '.cache_'+ model_id + '_' + split + '.json')
    best_cider=0
    #gdindex=[0,1,2,3,4]
    gdindex=[-1]
    cider_list =[]
    for i in gdindex:
        annFile='coco-caption/person_captions4eval_'+str(i)+'.json'
        print(annFile)
        coco = COCO(annFile)    
        valids = coco.getImgIds()

        # filter results to only those in MSCOCO validation set (will be about a third)
        preds_filt = [p for p in preds if p['image_id'] in valids]
        print('using %d/%d predictions' % (len(preds_filt), len(preds)))
        json.dump(preds_filt, open(cache_path, 'w')) # serialize to temporary json file. Sigh, COCO API...
        cocoRes = coco.loadRes(cache_path)
        cocoEval = COCOEvalCap(coco, cocoRes)
        cocoEval.params['image_id'] = cocoRes.getImgIds()
        cocoEval.evaluate()
        cider_list.append(cocoEval.eval['CIDEr'])
        # create output dictionary
        if cocoEval.eval['CIDEr']>=best_cider:
            best_cider = cocoEval.eval['CIDEr']
            out = {}
            for metric, score in cocoEval.eval.items():
                out[metric] = score

            imgToEval = cocoEval.imgToEval
                # collect SPICE_sub_score
            #for k in imgToEval.values()[0]['SPICE'].keys():
            #    if k != 'All':
            #        out['SPICE_'+k] = np.array([v['SPICE'][k]['f'] for v in  imgToEval.values()])
            #        out['SPICE_'+k] = (out['SPICE_'+k][out['SPICE_'+k]==out['SPICE_'+k]]).mean()
            
            for p in preds_filt:
                image_id, caption = p['image_id'], p['caption']
                imgToEval[image_id]['caption'] = caption
            #update predictions
            for i in range(len(preds)):
                if preds[i]['image_id'] in imgToEval:
                    preds[i]['eval'] = imgToEval[preds[i]['image_id']]

            out['bad_count_rate'] = sum([count_bad(_['caption']) for _ in preds_filt]) / float(len(preds_filt))
        else:
            continue
    outfile_path = os.path.join('eval_results/', model_id + '_' + split + '.json')
    with open(outfile_path, 'w') as outfile:
        json.dump({'overall': out, 'imgToEval': imgToEval}, outfile)
    cider_list=np.array(cider_list)
    print("min:",np.min(cider_list)," max:",np.max(cider_list)," mean:",np.mean(cider_list)," std:",np.std(cider_list))
    return out

def eval_split(model, crit, loader, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)
    verbose_beam = eval_kwargs.get('verbose_beam', 1)
    verbose_loss = eval_kwargs.get('verbose_loss', 1)
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'val')
    lang_eval = eval_kwargs.get('language_eval', 0)
    rank_eval = eval_kwargs.get('rank_eval', 0)
    dataset = eval_kwargs.get('dataset', 'person')
    beam_size = eval_kwargs.get('beam_size', 1)
    remove_bad_endings = eval_kwargs.get('remove_bad_endings', 0)
    os.environ["REMOVE_BAD_ENDINGS"] = str(remove_bad_endings) # Use this nasty way to make other code clean since it's a global configuration
    use_joint=eval_kwargs.get('use_joint', 0)
    init_scorer('person-'+split+'-words')
    # Make sure in the evaluation mode
    model.eval()

    loader.reset_iterator(split)

    n = 0
    loss = 0
    losses={}
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    visual={"image_id":[],"personality":[],"generation":[],"gd":[],"densecap":[],"Bleu1_gen/cap":[],"Bleu2_gen/cap":[],"Bleu3_gen/cap":[],"Bleu4_gen/cap":[],"Cider_gen/cap":[],"Bleu1_gen/gd":[],"Bleu2_gen/gd":[],"Bleu3_gen/gd":[],"Bleu4_gen/gd":[],"Cider_gen/gd":[],"Bleu1_cap/gd":[],"Bleu2_cap/gd":[],"Bleu3_cap/gd":[],"Bleu4_cap/gd":[],"Cider_cap/gd":[], "Bleu1_gd/gen":[],"Bleu2_gd/gen":[],"Bleu3_gd/gen":[],"Bleu4_gd/gen":[],"Cider_gd/gen":[]}
    if split=='change':
        visual['new_personality']=[]
    minopt=0
    verbose_loss = True
    while True:
        data = loader.get_batch(split)
        n = n + loader.batch_size
        if data.get('labels', None) is not  None and verbose_loss:
            # forward the model to get loss
            tmp = [data['fc_feats'], data['att_feats'],data['densecap'], data['labels'], data['masks'], data['att_masks'], data['personality']]
            tmp = [_.cuda() if _ is not None else _ for _ in tmp]
            fc_feats, att_feats,densecap, labels, masks, att_masks,personality = tmp
            with torch.no_grad():
               if eval_kwargs.get("use_dl",0)>0:
                    gen_result, sample_logprobs,alogprobs  = model(fc_feats, att_feats,densecap, att_masks,personality, opt={'sample_method':'sample'}, mode='sample')
                    loss = crit(model(fc_feats, att_feats,densecap, labels, att_masks,personality), alogprobs, labels[:,1:], masks[:,1:]).item()
               else:
                   loss = crit(model(fc_feats, att_feats,densecap, labels, att_masks,personality), labels[:,1:], masks[:,1:])
            
            loss_sum = loss_sum + loss
            loss_evals = loss_evals + 1
            if use_joint==1:
                for k,v in model.loss().items():
                    if k not in losses:
                        losses[k] = 0
                    losses[k] += v
        # forward the model to also get generated samples for each image
        # Only leave one feature for each image, in case duplicate sample
        tmp = [data['fc_feats'][np.arange(loader.batch_size)], 
            data['att_feats'][np.arange(loader.batch_size)] if data['att_feats'] is not None else None,
            data['densecap'][np.arange(loader.batch_size)],
            data['att_masks'][np.arange(loader.batch_size)] if data['att_masks'] is not None else None,
            data['personality'][np.arange(loader.batch_size)]]
        tmp = [_.cuda() if _ is not None else _ for _ in tmp]
        fc_feats, att_feats,densecap, att_masks,personality = tmp
        if split =='change':
            for pindex,pid in personality.nonzero():
                personality[pindex][pid]=0
                newpid = random.choice(range(1,len(personality)-1))
                personality[pindex][newpid]=1
        ground_truth =  data['labels'][:][:,1:]
        # forward the model to also get generated samples for each image
        with torch.no_grad():
            seq = model(fc_feats, att_feats,densecap, att_masks,personality, opt=eval_kwargs, mode='sample')[0].data
        
        # Print beam search
#        if beam_size > 1 and verbose_beam:
#            for i in range(loader.batch_size):
#                print('\n'.join([utils.decode_sequence(loader.get_vocab(), _['seq'].unsqueeze(0))[0] for _ in model.done_beams[i]]))
#                print('--' * 10)
        sents = utils.decode_sequence(loader.get_vocab(), seq)
        gd_display = utils.decode_sequence(loader.get_vocab(), ground_truth)
        for k, s in enumerate(sents):
            if beam_size > 1 and verbose_beam:
                beam_sents = [utils.decode_sequence(loader.get_vocab(), _['seq'].unsqueeze(0))[0] for _ in model.done_beams[k]] 
                maxcider=0
                mincider=1000
                sent =s
                for b,sq in enumerate(beam_sents):
                    current_cider=cal_cider(gd_display[k*loader.seq_per_img:(k+1)*loader.seq_per_img],sq)
                    if current_cider >= maxcider:
                        maxcider=current_cider
                        sentmax=sq
                    if current_cider <= mincider:
                        mincider=current_cider
                        sentmin=sq
                    if minopt==1:
                        sent=sentmin
                    elif minopt==-1:
                        sent=sentmax
                    else:
                        sent=s
                    
            else:
                sent = s
            #print("best sentence: ",sent) 
            newpidstr = str(personality[k].nonzero()[0].item())
            changed_personality =loader.get_personality()[newpidstr]
            entry = {'image_id': data['infos'][k]['id']+"_"+data['infos'][k]['personality'], 'caption':sent,'gd':gd_display[k*loader.seq_per_img:(k+1)*loader.seq_per_img]}
            if( entry not in predictions ):
                densecap_display = utils.decode_sequence(loader.get_vocab(), data['densecap'][k])
                allscore = get_scores_separate([densecap_display],[sent]) # gd is the densecap and test is generation, len(common)/len(generation)
                for bk in allscore:
                    visual[bk+"_gen/cap"].append(allscore[bk])
                allscore_gd = get_scores_separate([gd_display[k*loader.seq_per_img:(k+1)*loader.seq_per_img]],[sent])
                for bkgd in allscore_gd:
                    visual[bkgd+"_gen/gd"].append(allscore_gd[bkgd])
                allscore_capgd = get_scores_separate([gd_display[k*loader.seq_per_img:(k+1)*loader.seq_per_img]],densecap_display)
                for cap_bkgd in allscore_capgd:
                    visual[cap_bkgd+"_cap/gd"].append(allscore_capgd[cap_bkgd])
                
                allscore_gd_flip = get_scores_separate([[sent]],gd_display[k*loader.seq_per_img:(k+1)*loader.seq_per_img]) 
                for bkgd in allscore_gd_flip:
                    visual[bkgd+"_gd/gen"].append(allscore_gd_flip[bkgd])                
                
                visual["image_id"].append(data['infos'][k]['id'])
                visual["personality"].append(data['infos'][k]['personality'])
                if split=='change':
                    visual["new_personality"].append(changed_personality)
                visual['generation'].append(sent)
                visual["gd"].append(gd_display[k*loader.seq_per_img:(k+1)*loader.seq_per_img])
                visual["densecap"].append(densecap_display)
            if eval_kwargs.get('dump_path', 0) == 1:
                entry['file_name'] = data['infos'][k]['file_path']
            predictions.append(entry)
            if eval_kwargs.get('dump_images', 0) == 1:
                # dump the raw image to vis/ folder
                cmd = 'cp "' + os.path.join(eval_kwargs['image_root'], data['infos'][k]['file_path']) + '" vis/imgs/img' + str(len(predictions)) + '.jpg' # bit gross
                print(cmd)
                os.system(cmd)

            if verbose:
                print('--------------------------------------------------------------------')
                if split=='change':
                    print('image %s{%s--------->%s}: %s' %(entry['image_id'],changed_personality,entry['gd'], entry['caption']))
                else:
                    print('image %s{%s}: %s' %(entry['image_id'],entry['gd'], entry['caption']))
                print('--------------------------------------------------------------------')

        # if we wrapped around the split or used up val imgs budget then bail
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        for i in range(n - ix1):
            predictions.pop()
        if verbose:
            print('evaluating validation preformance... %d/%d (%f)' %(ix0 - 1, ix1, loss))
        if data['bounds']['wrapped']:
            break
        if num_images >= 0 and n >= num_images:
            break
    allwords = " ".join(visual['generation'])
    allwords = allwords.split(" ")
    print("sets length of allwords:",len(set(allwords)))
    print("length of allwords:",len(allwords))
    print("rate of set/all:",len(set(allwords))/len(allwords))
    lang_stats = None
    if lang_eval == 1:
        lang_stats = language_eval(dataset, predictions, eval_kwargs['id'], split)
    
    df = pd.DataFrame.from_dict(visual)
    df.to_csv("visual_res/"+eval_kwargs['id']+"_"+str(split)+"_"+"visual.csv")
    if use_joint==1:
        ranks = evalrank(model, loader, eval_kwargs) if rank_eval else {}
    # Switch back to training mode

    model.train()
    if use_joint==1:
        losses = {k:v/loss_evals for k,v in losses.items()}
        losses.update(ranks)
        return losses, predictions, lang_stats
    return loss_sum/loss_evals, predictions, lang_stats


def encode_data(model, loader, eval_kwargs={}):
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'val')
    dataset = eval_kwargs.get('dataset', 'coco')

    # Make sure in the evaluation mode
    model.eval()

    loader_seq_per_img = loader.seq_per_img
    loader.seq_per_img = 5
    loader.reset_iterator(split)

    n = 0
    img_embs = []
    cap_embs = []
    while True:
        data = loader.get_batch(split)
        n = n + loader.batch_size

        tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks']]
        tmp = utils.var_wrapper(tmp)
        fc_feats, att_feats, labels, masks = tmp

        with torch.no_grad():
            img_emb = model.vse.img_enc(fc_feats)
            cap_emb = model.vse.txt_enc(labels, masks)

        # if we wrapped around the split or used up val imgs budget then bail
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)

        if n > ix1:
            img_emb = img_emb[:(ix1-n)*loader.seq_per_img]
            cap_emb = cap_emb[:(ix1-n)*loader.seq_per_img]

        # preserve the embeddings by copying from gpu and converting to np
        img_embs.append(img_emb.data.cpu().numpy().copy())
        cap_embs.append(cap_emb.data.cpu().numpy().copy())

        if data['bounds']['wrapped']:
            break
        if num_images >= 0 and n >= num_images:
            break

        print("%d/%d"%(n,ix1))

    img_embs = np.vstack(img_embs)
    cap_embs = np.vstack(cap_embs)

    assert img_embs.shape[0] == ix1 * loader.seq_per_img

    loader.seq_per_img = loader_seq_per_img

    return img_embs, cap_embs


def evalrank(model, loader, eval_kwargs={}):
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'val')
    dataset = eval_kwargs.get('dataset', 'coco')
    fold5 = eval_kwargs.get('fold5', 0)
    """
    Evaluate a trained model on either dev or test. If `fold5=True`, 5 fold
    cross-validation is done (only for MSCOCO). Otherwise, the full data is
    used for evaluation.
    """
    print('Computing results...')
    img_embs, cap_embs = encode_data(model, loader, eval_kwargs)
    print('Images: %d, Captions: %d' %
          (img_embs.shape[0] / 5, cap_embs.shape[0]))

    if not fold5:
        # no cross-validation, full evaluation
        r, rt = i2t(img_embs, cap_embs, measure='cosine', return_ranks=True)
        ri, rti = t2i(img_embs, cap_embs,
                      measure='cosine', return_ranks=True)
        ar = (r[0] + r[1] + r[2]) / 3
        ari = (ri[0] + ri[1] + ri[2]) / 3
        rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
        print("rsum: %.1f" % rsum)
        print("Average i2t Recall: %.1f" % ar)
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
        print("Average t2i Recall: %.1f" % ari)
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)
    else:
        # 5fold cross-validation, only for MSCOCO
        results = []
        for i in range(5):
            r, rt0 = i2t(img_embs[i * 5000:(i + 1) * 5000],
                         cap_embs[i * 5000:(i + 1) *
                                  5000], measure='cosine',
                         return_ranks=True)
            print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % r)
            ri, rti0 = t2i(img_embs[i * 5000:(i + 1) * 5000],
                           cap_embs[i * 5000:(i + 1) *
                                    5000], measure='cosine',
                           return_ranks=True)
            if i == 0:
                rt, rti = rt0, rti0
            print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % ri)
            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            print("rsum: %.1f ar: %.1f ari: %.1f" % (rsum, ar, ari))
            results += [list(r) + list(ri) + [ar, ari, rsum]]

        print("-----------------------------------")
        print("Mean metrics: ")
        mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
        print("rsum: %.1f" % (mean_metrics[10] * 6))
        print("Average i2t Recall: %.1f" % mean_metrics[11])
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[:5])
        print("Average t2i Recall: %.1f" % mean_metrics[12])
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[5:10])

    return {'rsum':rsum, 'i2t_ar':ar, 't2i_ar':ari,
            'i2t_r1':r[0], 'i2t_r5':r[1], 'i2t_r10':r[2], 'i2t_medr':r[3], 'i2t_meanr':r[4],
            't2i_r1':ri[0], 't2i_r5':ri[1], 't2i_r10':ri[2], 't2i_medr':ri[3], 't2i_meanr':ri[4]}#{'rt': rt, 'rti': rti}


def i2t(images, captions, npts=None, measure='cosine', return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts is None:
        npts = images.shape[0] // 5
    index_list = []

    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    for index in range(npts):

        # Get query image
        im = images[5 * index].reshape(1, images.shape[1])

        # Compute scores
        if measure == 'order':
            bs = 100
            if index % bs == 0:
                mx = min(images.shape[0], 5 * (index + bs))
                im2 = images[5 * index:mx:5]
                d2 = order_sim(torch.Tensor(im2).cuda(),
                               torch.Tensor(captions).cuda())
                d2 = d2.cpu().numpy()
            d = d2[index % bs]
        else:
            d = np.dot(im, captions.T).flatten()
        inds = np.argsort(d)[::-1]
        index_list.append(inds[0])

        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(images, captions, npts=None, measure='cosine', return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts is None:
        npts = images.shape[0] // 5
    ims = np.array([images[i] for i in range(0, len(images), 5)])

    ranks = np.zeros(5 * npts)
    top1 = np.zeros(5 * npts)
    for index in range(npts):

        # Get query captions
        queries = captions[5 * index:5 * index + 5]

        # Compute scores
        if measure == 'order':
            bs = 100
            if 5 * index % bs == 0:
                mx = min(captions.shape[0], 5 * index + bs)
                q2 = captions[5 * index:mx]
                d2 = order_sim(torch.Tensor(ims).cuda(),
                               torch.Tensor(q2).cuda())
                d2 = d2.cpu().numpy()

            d = d2[:, (5 * index) % bs:(5 * index) % bs + 5].T
        else:
            d = np.dot(queries, ims.T)
        inds = np.zeros(d.shape)
        for i in range(len(inds)):
            inds[i] = np.argsort(d[i])[::-1]
            ranks[5 * index + i] = np.where(inds[i] == index)[0][0]
            top1[5 * index + i] = inds[i][0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)
