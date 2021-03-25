# This file contains Att2in2, AdaAtt, AdaAttMO, TopDown model

# AdaAtt is from Knowing When to Look: Adaptive Attention via A Visual Sentinel for Image Captioning
# https://arxiv.org/abs/1612.01887
# AdaAttMO is a modified version with maxout lstm

# Att2in is from Self-critical Sequence Training for Image Captioning
# https://arxiv.org/abs/1612.00563
# In this file we only have Att2in2, which is a slightly different version of att2in,
# in which the img feature embedding and word embedding is the same as what in adaatt.

# TopDown is from Bottom-Up and Top-Down Attention for Image Captioning and VQA
# https://arxiv.org/abs/1707.07998
# However, it may not be identical to the author's architecture.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import misc.utils as utils
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

from torch.autograd import Variable
from .denseCaptionModel import CaptionModel

bad_endings = ['a','an','the','in','for','at','of','with','before','after','on','upon','near','to','is','are','am']
bad_endings += ['the']

def sort_pack_padded_sequence(input, lengths):
    sorted_lengths, indices = torch.sort(lengths, descending=True)
    tmp = pack_padded_sequence(input[indices], sorted_lengths, batch_first=True)
    inv_ix = indices.clone()
    inv_ix[indices] = torch.arange(0,len(indices)).type_as(inv_ix)
    return tmp, inv_ix

def pad_unsort_packed_sequence(input, inv_ix):
    tmp, _ = pad_packed_sequence(input, batch_first=True)
    tmp = tmp[inv_ix]
    return tmp

def pack_wrapper(module, att_feats, att_masks):
    if att_masks is not None:
        packed, inv_ix = sort_pack_padded_sequence(att_feats, att_masks.data.long().sum(1))
        return pad_unsort_packed_sequence(PackedSequence(module(packed[0]), packed[1]), inv_ix)
    else:
        return module(att_feats)

class densepembedAttModel(CaptionModel):
    def __init__(self, opt):
        super(densepembedAttModel, self).__init__()
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        #self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.batch_size = opt.batch_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = getattr(opt, 'max_length', 16) or opt.seq_length # maximum sample length
        self.seq_length = opt.seq_length 
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size
        self.use_bn = getattr(opt, 'use_bn', 0)
        self.bos_idx = getattr(opt, 'bos_idx', 0)
        self.eos_idx = getattr(opt, 'eos_idx', 0)
        self.pad_idx = getattr(opt, 'pad_idx', 0)
        self.perss_hidden_dim = opt.input_encoding_size # same size as text encoding
        self.num_personalities = len(opt.xpersonality)+1
        self.perss_dropout = opt.drop_prob_lm #same drop out of personality embedding as text
        self.ss_prob = 0.0 # Schedule sampling probability

        self.embed = nn.Sequential(nn.Embedding(self.vocab_size + 1, self.input_encoding_size),
                                nn.ReLU(),
                                nn.Dropout(self.drop_prob_lm))
        self.fc_embed = nn.Sequential(nn.Linear(self.fc_feat_size, self.rnn_size),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))
        self.att_embed = nn.Sequential(*(
                                    ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ())+  # norm1d or 2d?lcx
                                    (nn.Linear(self.att_feat_size, self.rnn_size),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))+
                                    ((nn.BatchNorm1d(self.rnn_size),) if self.use_bn==2 else ())))
        self._build_personality_encoder()
        self.encoder=nn.LSTMCell(self.input_encoding_size, opt.rnn_size)
        self.logit_layers = getattr(opt, 'logit_layers', 1)
        if self.logit_layers == 1:
            self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)
        else:
            self.logit = [[nn.Linear(self.rnn_size, self.rnn_size), nn.ReLU(), nn.Dropout(0.5)] for _ in range(opt.logit_layers - 1)]
            self.logit = nn.Sequential(*(reduce(lambda x,y:x+y, self.logit) + [nn.Linear(self.rnn_size, self.vocab_size + 1)]))
        self.ctx2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.ctx2att_t = nn.Linear(self.rnn_size, self.att_hid_size)
        # For remove bad endding
        self.vocab = opt.vocab
        self.bad_endings_ix = [int(k) for k,v in self.vocab.items() if v in bad_endings]
        self.decoding_constraint = getattr(opt, 'decoding_constraint', 0)

        self._loss = {'xe':0}
    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, bsz, self.rnn_size),
                weight.new_zeros(self.num_layers, bsz, self.rnn_size))
    def enc_init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(bsz, self.rnn_size),
                weight.new_zeros(bsz, self.rnn_size))
    def clip_att(self, att_feats, att_masks):
        # Clip the length of att_masks and att_feats to the maximum length
        if att_masks is not None:
            max_len = att_masks.data.long().sum(1).max()
            att_feats = att_feats[:, :max_len].contiguous()
            att_masks = att_masks[:, :max_len].contiguous()
        return att_feats, att_masks

    def _prepare_feature(self, fc_feats, att_feats, att_masks):  
        att_feats, att_masks = self.clip_att(att_feats, att_masks)
        # embed fc and att feats

        fc_feats = self.fc_embed(torch.squeeze(fc_feats))
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks) #batchsize*49*Rnn-size
        # Project the attention feats first to reduce memory and computation comsumptions.
        p_att_feats = self.ctx2att(att_feats)  #change from rnnsize to batchsize*49*att_hid_size(512)
        return fc_feats, att_feats, p_att_feats, att_masks

    def _build_personality_encoder(self):
        personality_layers = [
            nn.Embedding(self.num_personalities+1, self.input_encoding_size),
            nn.Linear(self.input_encoding_size, self.perss_hidden_dim),
        ]
        self.personality_encoder = nn.Sequential(*personality_layers)

    def _forward(self, ofc_feats, oatt_feats,densecap, seq, att_masks=None,personality=None):
        batch_size = self.batch_size
        seq_per_img = seq.shape[0] // batch_size
        outputs = torch.zeros(batch_size*seq_per_img, seq.size(1) - 1, self.vocab_size+1,dtype=torch.float).cuda()
        # Prepare the features
        rp_fc_feats, rp_att_feats, rpp_att_feats, rp_att_masks = self._prepare_feature(ofc_feats, oatt_feats,att_masks)
        # pp_att_feats is used for attention, we cache it in advance to reduce computation cost
        encodestate = self.enc_init_hidden(batch_size*5)
        encoder_cells =[]
        for k in range(densecap.size(-1)):
            w =  densecap[:,:,k].clone()
            embedw = self.embed(w)
            embedw = embedw.contiguous().view(-1,embedw.size(-1)).contiguous()
            encodestate= self.encoder(embedw, (encodestate[0],encodestate[1])) 
            encoder_cells.append(encodestate[1].contiguous().view(batch_size,5,encodestate[1].size(-1)))       
        hstate, cstate = encodestate
        att_feats = torch.stack(encoder_cells).cuda()
        att_feats = att_feats.contiguous().permute(1,2,0,3)

        fc_feats =  hstate.contiguous().view(batch_size,5,encodestate[0].size(-1))
        fc_feats =  fc_feats.contiguous().view(batch_size,-1) 
        p_att_feats = self.ctx2att_t(att_feats)
        decodestate = self.init_hidden(batch_size*seq_per_img)
        if seq_per_img > 1:
            fc_feats, att_feats, p_att_feats, att_masks = utils.repeat_tensors(seq_per_img,
                    [fc_feats, att_feats, p_att_feats, att_masks])
            rp_fc_feats, rp_att_feats, rpp_att_feats, rp_att_masks = utils.repeat_tensors(seq_per_img,[rp_fc_feats, rp_att_feats, rpp_att_feats, rp_att_masks])

        for i in range(seq.size(1) - 1):
            if self.training and i >= 1 and self.ss_prob > 0.0: # otherwiste no need to sample
                sample_prob = fc_feats.new(batch_size*seq_per_img).uniform_(0, 1)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    it = seq[:, i].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = seq[:, i].data.clone()
                    #prob_prev = torch.exp(outputs[-1].data.index_select(0, sample_ind)) # fetch prev distribution: shape Nx(M+1)
                    #it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1))
                    # prob_prev = torch.exp(outputs[-1].data) # fetch prev distribution: shape Nx(M+1)
                    prob_prev = torch.exp(outputs[:, i-1].detach()) # fetch prev distribution: shape Nx(M+1)
                    it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
            else:
                it = seq[:, i].clone()          
            # break if all the sequences end
            if i >= 1 and seq[:, i].sum() == 0:
                break
             
            output,  decodestate = self.get_logprobs_state(it,personality, fc_feats, att_feats, p_att_feats, att_masks,rp_fc_feats,               rp_att_feats, rpp_att_feats, rp_att_masks, decodestate)
            outputs[:, i] = output
        return outputs
    
    def get_logprobs_state(self, it,personality, fc_feats, att_feats, p_att_feats, att_masks,rp_fc_feats,rp_att_feats, rpp_att_feats, rp_att_masks, state):
        # 'it' contains a word index
        batch_size = personality.size(0)
        xt = self.embed(it)# 500*100
        seq_per_img = xt.size(0)//batch_size
        if personality is not None:
            pers_encoded = self.personality_encoder(personality.nonzero(as_tuple=True)[1])
            pers_encoded = utils.repeat_tensors(seq_per_img,pers_encoded)
            xt=torch.cat((xt,pers_encoded),1)
        output, state = self.core(xt, fc_feats, att_feats, p_att_feats, state, att_masks,rp_fc_feats,rp_att_feats, rpp_att_feats, rp_att_masks)
        logitoutput = self.logit(output)
        finallogprobs = F.log_softmax(logitoutput, dim=1)

        return finallogprobs, state

    def _sample_beam(self, ofc_feats, oatt_feats, densecap, att_masks=None,personality=None, opt={}):
        beam_size = opt.get('beam_size', 10)
        batch_size = densecap.size(0) 
        rp_fc_feats, rp_att_feats, rpp_att_feats, rp_att_masks = self._prepare_feature(ofc_feats, oatt_feats, att_masks)
        outputs = torch.zeros(batch_size, self.seq_length + 1, self.vocab_size+1,dtype=torch.float).cuda()
        # Prepare the features
        #p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats,densecap, att_masks)
        # pp_att_feats is used for attention, we cache it in advance to reduce computation cost
        encodestate = self.enc_init_hidden(batch_size*5)
        encoder_cells =[]
        for k in range(densecap.size(-1)):
            w =  densecap[:,:,k].clone()
            embedw = self.embed(w)
            embedw = embedw.contiguous().view(-1,embedw.size(-1)).contiguous()
            encodestate= self.encoder(embedw, (encodestate[0],encodestate[1]))
            encoder_cells.append(encodestate[1].contiguous().view(batch_size,5,encodestate[1].size(-1)))
        hstate, cstate = encodestate
        att_feats = torch.stack(encoder_cells).cuda()
        p_att_feats = att_feats.contiguous().permute(1,2,0,3)

        fc_feats =  hstate.contiguous().view(batch_size,5,encodestate[0].size(-1))
        p_fc_feats =  fc_feats.contiguous().view(batch_size,-1)
        pp_att_feats = self.ctx2att_t(p_att_feats)
        p_att_masks  = att_masks



        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)
            tmp_fc_feats = p_fc_feats[k:k+1].expand(beam_size, p_fc_feats.size(1))
            tmp_att_feats = p_att_feats[k:k+1].expand(*((beam_size,)+p_att_feats.size()[1:])).contiguous()
            tmp_p_att_feats = pp_att_feats[k:k+1].expand(*((beam_size,)+pp_att_feats.size()[1:])).contiguous()
            tmp_att_masks = p_att_masks[k:k+1].expand(*((beam_size,)+p_att_masks.size()[1:])).contiguous() if att_masks is not None else None
            tmp_rp_fc_feats = rp_fc_feats[k:k+1].expand(*((beam_size,)+rp_fc_feats.size()[1:])).contiguous()
            tmp_rp_att_feats = rp_att_feats[k:k+1].expand(*((beam_size,)+rp_att_feats.size()[1:])).contiguous()
            tmp_rpp_att_feats = rpp_att_feats[k:k+1].expand(*((beam_size,)+rpp_att_feats.size()[1:])).contiguous()
            tmp_rp_att_masks = rp_att_masks[k:k+1].expand(*((beam_size,)+rp_att_masks.size()[1:])).contiguous() if rp_att_masks is not None else None
            tmp_personality=personality[k:k+1].expand(*((beam_size,)+personality.size()[1:])).contiguous()
            for t in range(1):
                if t == 0: # input <bos>
                    it = fc_feats.new_zeros([beam_size], dtype=torch.long)
                logprobs, state = self.get_logprobs_state(it, tmp_personality,tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, tmp_att_masks,tmp_rp_fc_feats, tmp_rp_att_feats, tmp_rpp_att_feats, tmp_rp_att_masks,  state)
            self.done_beams[k] = self.beam_search(state, logprobs,tmp_personality,tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, tmp_att_masks,tmp_rp_fc_feats, tmp_rp_att_feats, tmp_rpp_att_feats, tmp_rp_att_masks, opt=opt)
            seq[:, k] = self.done_beams[k][0]['seq'] # the first beam has highest cumulative score
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

    def _sample(self, ofc_feats, oatt_feats,densecap, att_masks=None,personality=None, opt={}): # get softmax everytime after constrain

        sample_method = opt.get('sample_method', 'greedy')
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        opt['block_trigrams'] =0
        opt['remove_bad_endings'] =1
        opt['decoding_constraint'] =1
        decoding_constraint = opt.get('decoding_constraint', 1)
        block_trigrams = opt.get('block_trigrams', 1)
        remove_bad_endings = opt.get('remove_bad_endings', 1)
        no_unk=1
        if beam_size > 1:
            return self._sample_beam(ofc_feats, oatt_feats,densecap, att_masks,personality, opt)

        batch_size = densecap.size(0)
        # Prepare the features
        rp_fc_feats, rp_att_feats, rpp_att_feats, rp_att_masks = self._prepare_feature(ofc_feats, oatt_feats,att_masks)
        # pp_att_feats is used for attention, we cache it in advance to reduce computation cost
        encodestate = self.enc_init_hidden(batch_size*5)
        encoder_cells =[]
        for k in range(densecap.size(-1)):
            w =  densecap[:,:,k].clone()
            embedw = self.embed(w)
            embedw = embedw.contiguous().view(-1,embedw.size(-1)).contiguous()
            encodestate= self.encoder(embedw, (encodestate[0],encodestate[1]))
            encoder_cells.append(encodestate[1].contiguous().view(batch_size,5,encodestate[1].size(-1)))
        hstate, cstate = encodestate
        att_feats = torch.stack(encoder_cells).cuda()
        p_att_feats = att_feats.contiguous().permute(1,2,0,3)

        fc_feats =  hstate.contiguous().view(batch_size,5,encodestate[0].size(-1))
        p_fc_feats =  fc_feats.contiguous().view(batch_size,-1)
        pp_att_feats = self.ctx2att_t(p_att_feats)
        p_att_masks =  att_masks

        decodestate = self.init_hidden(batch_size)
      
        #p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        trigrams = [] # will be a list of batch_size dictionaries
        alogprobs1 = torch.zeros(batch_size,self.seq_length+1, self.vocab_size+1).cuda()
        alogprobs= torch.zeros(batch_size,self.seq_length+1, self.vocab_size+1).cuda()
        for bk in range(alogprobs.size(0)):
            alogprobs[bk]=nn.LogSoftmax(dim=1)(alogprobs1[bk])
        seq = torch.zeros((batch_size, self.seq_length), dtype=torch.long).cuda()
        seqLogprobs = torch.zeros(batch_size, self.seq_length,dtype=torch.float).cuda()
        for t in range(self.seq_length + 1):
            if t == 0: # input <bos>
                it = fc_feats.new_zeros(batch_size, dtype=torch.long)
            logprobs, decodestate = self.get_logprobs_state(it,personality, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks,rp_fc_feats,rp_att_feats, rpp_att_feats, rp_att_masks, decodestate)
            if no_unk==1: # change the unk constrain
                mask2 = torch.zeros(logprobs.size(), requires_grad=False).cuda()
                mask2[:,mask2.size(1)-1] =-10e20
                logprobs= logprobs+ mask2
                logprobs = F.log_softmax(logprobs,dim=-1) # after re-computer the log then constrain other things
            if decoding_constraint and t > 0:
                tmp = logprobs.new_zeros(logprobs.size())
                tmp.scatter_(1, seq[:,t-1].data.unsqueeze(1), float('-10e20'))
                logprobs = logprobs + tmp
                logprobs = F.log_softmax(logprobs,dim=-1) # after re-computer the log then constrain other things
            if remove_bad_endings and t > 0:
                tmp = logprobs.new_zeros(logprobs.size())
                prev_bad = np.isin(seq[:,t-1].data.cpu().numpy(), self.bad_endings_ix)
                # Impossible to generate remove_bad_endings
                tmp[torch.from_numpy(prev_bad.astype(np.bool_)), 0] = float('-10e20')
                logprobs = logprobs + tmp
                logprobs = F.log_softmax(logprobs,dim=-1) # after re-computer the log then constrain other things
            # Mess with trigrams
            if block_trigrams and t >= 3:
                # Store trigram generated at last step
                prev_two_batch = seq[:,t-3:t-1]
                for i in range(batch_size): # = seq.size(0)
                    prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                    current  = seq[i][t-1]
                    if t == 3: # initialize
                        trigrams.append({prev_two: [current]}) # {LongTensor: list containing 1 int}
                    elif t > 3:
                        if prev_two in trigrams[i]: # add to list
                            trigrams[i][prev_two].append(current)
                        else: # create list
                            trigrams[i][prev_two] = [current]
                # Block used trigrams at next step
                prev_two_batch = seq[:,t-2:t]
                mask = torch.zeros(logprobs.size(), requires_grad=False).cuda() # batch_size x vocab_size
                for i in range(batch_size):
                    prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                    if prev_two in trigrams[i]:
                        for j in trigrams[i][prev_two]:
                            mask[i,j] += 1
                # Apply mask to log probs
                #logprobs = logprobs - (mask * 1e9)
                alpha = 10e20 # = 4
                logprobs = logprobs + (mask * -0.693 * alpha) # ln(1/2) * alpha (alpha -> infty works best)
                logprobs = F.log_softmax(logprobs,dim=-1) # after re-computer the log then constrain other things
            # sample the next word
            if t == self.seq_length: # skip if we achieve maximum length
                break
            it, sampleLogprobs = self.sample_next_word(logprobs, sample_method, temperature)

            # stop when all finished
            if t == 0:
                unfinished = it > 0
            else:
                unfinished = unfinished * (it > 0)
            it = it * unfinished.type_as(it)
            seq[:,t] = it
            seqLogprobs[:,t] = sampleLogprobs.view(-1)
            # quit loop if all sequences have finished
            alogprobs[:, t] = logprobs
            if unfinished.sum() == 0:
                break

        return seq, seqLogprobs, alogprobs
#  create samplen
    def _samplen(self, ofc_feats, oatt_feats,densecap, att_masks=None,personality=None, opt={}):

        sample_method = opt.get('sample_method', 'greedy')
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        opt['block_trigrams'] =0
        opt['remove_bad_endings'] =1
        opt['decoding_constraint'] =1
        decoding_constraint = opt.get('decoding_constraint', 1)
        block_trigrams = opt.get('block_trigrams', 1)
        remove_bad_endings = opt.get('remove_bad_endings', 1)
        sample_n = int(opt.get('sample_n', 3))
        no_unk=1
        if beam_size > 1:
            return self._sample_beam(ofc_feats, oatt_feats,densecap, att_masks,personality, opt)

        batch_size = densecap.size(0)
        # Prepare the features
        rp_fc_feats, rp_att_feats, rpp_att_feats, rp_att_masks = self._prepare_feature(ofc_feats, oatt_feats,att_masks)
        # pp_att_feats is used for attention, we cache it in advance to reduce computation cost
        if sample_n > 1:
            personality, densecap, rp_fc_feats, rp_att_feats, rpp_att_feats, rp_att_masks = utils.repeat_tensors(sample_n,
                [personality,densecap,rp_fc_feats, rp_att_feats, rpp_att_feats, rp_att_masks]
            ) 
        encodestate = self.enc_init_hidden(batch_size*5*sample_n)
        encoder_cells =[]
        for k in range(densecap.size(-1)):
            w =  densecap[:,:,k].clone()
            embedw = self.embed(w)
            embedw = embedw.contiguous().view(-1,embedw.size(-1)).contiguous()
            encodestate= self.encoder(embedw, (encodestate[0],encodestate[1]))
            encoder_cells.append(encodestate[1].contiguous().view(batch_size*sample_n,5,encodestate[1].size(-1)))
        hstate, cstate = encodestate
        att_feats = torch.stack(encoder_cells).cuda()
        p_att_feats = att_feats.contiguous().permute(1,2,0,3)

        fc_feats =  hstate.contiguous().view(batch_size*sample_n,5,encodestate[0].size(-1))
        p_fc_feats =  fc_feats.contiguous().view(batch_size*sample_n,-1)
        pp_att_feats = self.ctx2att_t(p_att_feats)
        p_att_masks =  att_masks

        decodestate = self.init_hidden(batch_size*sample_n)
      
        #p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        trigrams = [] # will be a list of batch_size dictionaries
        alogprobs1 = torch.zeros(batch_size*sample_n,self.seq_length+1, self.vocab_size+1).cuda()
        alogprobs= torch.zeros(batch_size*sample_n,self.seq_length+1, self.vocab_size+1).cuda()
        for bk in range(alogprobs.size(0)):
            alogprobs[bk]=nn.LogSoftmax(dim=1)(alogprobs1[bk])
        
        seq = fc_feats.new_full((batch_size*sample_n, self.seq_length), self.pad_idx, dtype=torch.long)
        seqLogprobs = torch.zeros(batch_size*sample_n, self.seq_length,dtype=torch.float).cuda()
        for t in range(self.seq_length + 1):
            if t == 0: # input <bos>
                it = fc_feats.new_zeros(batch_size*sample_n, dtype=torch.long)
            logprobs, decodestate = self.get_logprobs_state(it,personality, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks,rp_fc_feats,rp_att_feats, rpp_att_feats, rp_att_masks, decodestate)
            if decoding_constraint and t > 0:
                tmp = logprobs.new_zeros(logprobs.size())
                tmp.scatter_(1, seq[:,t-1].data.unsqueeze(1), float('-10e20'))
                logprobs = logprobs + tmp

            if remove_bad_endings and t > 0:
                tmp = logprobs.new_zeros(logprobs.size())
                prev_bad = np.isin(seq[:,t-1].data.cpu().numpy(), self.bad_endings_ix)
                # Impossible to generate remove_bad_endings
                tmp[torch.from_numpy(prev_bad.astype(np.bool_)), 0] = float('-10e20')
                logprobs = logprobs + tmp

            # Mess with trigrams
            if block_trigrams and t >= 3:
                # Store trigram generated at last step
                prev_two_batch = seq[:,t-3:t-1]
                for i in range(batch_size): # = seq.size(0)
                    prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                    current  = seq[i][t-1]
                    if t == 3: # initialize
                        trigrams.append({prev_two: [current]}) # {LongTensor: list containing 1 int}
                    elif t > 3:
                        if prev_two in trigrams[i]: # add to list
                            trigrams[i][prev_two].append(current)
                        else: # create list
                            trigrams[i][prev_two] = [current]
                # Block used trigrams at next step
                prev_two_batch = seq[:,t-2:t]
                mask = torch.zeros(logprobs.size(), requires_grad=False).cuda() # batch_size x vocab_size
                for i in range(batch_size):
                    prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                    if prev_two in trigrams[i]:
                        for j in trigrams[i][prev_two]:
                            mask[i,j] += 1
                # Apply mask to log probs
                #logprobs = logprobs - (mask * 1e9)
                alpha = 10e20 # = 4
                logprobs = logprobs + (mask * -0.693 * alpha) # ln(1/2) * alpha (alpha -> infty works best)
            if no_unk==1:
                mask2 = torch.zeros(logprobs.size(), requires_grad=False).cuda()
                mask2[:,mask2.size(1)-1] =-10e20
                logprobs= logprobs+ mask2
            logprobs = F.log_softmax(logprobs,dim=-1)
            # sample the next word
            if t == self.seq_length: # skip if we achieve maximum length
                break
            
            it, sampleLogprobs = self.sample_next_word(logprobs, sample_method, temperature)

            # stop when all finished
            if t == 0:
                unfinished = it > 0
            else:
                unfinished = unfinished * (it > 0)
            it = it * unfinished.type_as(it)
            seq[:,t] = it
            seqLogprobs[:,t] = sampleLogprobs.view(-1)
            # quit loop if all sequences have finished
            alogprobs[:, t] = logprobs
            if unfinished.sum() == 0:
                break

        return seq, seqLogprobs, alogprobs


class TopDownCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(TopDownCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm

        self.att_lstm = nn.LSTMCell(14*opt.input_encoding_size, opt.rnn_size) # we, fc, h^2_t-1
        self.ratt_lstm = nn.LSTMCell(2*opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size)
        self.lang_lstm = nn.LSTMCell(opt.rnn_size * 2, opt.rnn_size) # h^1_t, \hat v
        self.rlang_lstm = nn.LSTMCell(opt.rnn_size * 2, opt.rnn_size)
        self.attention = Attention(opt)
        self.attention2 = Attention(opt)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks,rp_fc_feats, rp_att_feats, rpp_att_feats, rp_att_masks):
        prev_h = state[0][-1] #rnn_size
        att_lstm_input = torch.cat([prev_h, fc_feats, xt], 1) # batch_size*(4*rnn_size)
        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))#(state[0][0],state[1][0])bothare100*1000, means (hx,cx) 
        att = self.attention(h_att, att_feats, p_att_feats, att_masks)
        # for resnext feature
        ratt_lstm_input = torch.cat([prev_h, rp_fc_feats, xt], 1) # batch_size*(4*rnn_size)
        rh_att, rc_att = self.ratt_lstm(ratt_lstm_input, (state[0][0], state[1][0]))#(state[0][0],state[1][0])bothare100*1000, means (hx,cx)
        ratt = self.attention2(rh_att, rp_att_feats, rpp_att_feats, rp_att_masks)
        
        lang_lstm_input = torch.cat([att, h_att], 1)
        # lang_lstm_input = torch.cat([att, F.dropout(h_att, self.drop_prob_lm, self.training)], 1) 
        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))

       # resnext features for language model
        rlang_lstm_input = torch.cat([ratt, rh_att], 1)
        rh_lang, rc_lang = self.rlang_lstm(rlang_lstm_input, (state[0][1], state[1][1]))
        
        output = F.dropout(h_lang, self.drop_prob_lm, self.training)
        routput = F.dropout(rh_lang, self.drop_prob_lm, self.training)
        w1= Variable(torch.ones_like(output), requires_grad=True)
        w2= Variable(torch.ones_like(routput), requires_grad=True)
        finaloutput = w1 *output + w2 * routput
        finalh_att = w1*h_att + w2 * rh_att
        finalc_att = w1*c_att + w2 * rc_att
        finalh_lang = w1*h_lang + w2*rh_lang
        finalc_lang = w1*c_lang + w2*rc_lang
        state = (torch.stack([finalh_att, finalh_lang]), torch.stack([finalc_att, finalc_lang]))        

        return finaloutput, state





class Attention(nn.Module):
    def __init__(self, opt):
        super(Attention, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size

        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)

    def forward(self, h, att_feats, p_att_feats, att_masks=None):
        # The p_att_feats here is already projected
        att_size = att_feats.numel() // att_feats.size(0) // att_feats.size(-1) #4900000
        att = p_att_feats.view(-1, att_size, self.att_hid_size) #[100, 49, 512]
        
        att_h = self.h2att(h)                        #change from batch*rnn_size to batch * att_hid_size
        att_h = att_h.unsqueeze(1).expand_as(att)            # batch * att_size * att_hid_size
        dot = att + att_h                                   # batch * att_size * att_hid_size
        dot = torch.tanh(dot)                                # batch * att_size * att_hid_size
        dot = dot.view(-1, self.att_hid_size)               # (batch * att_size) * att_hid_size [4900, 512]
        dot = self.alpha_net(dot)                           # (batch * att_size) * 1
        dot = dot.view(-1, att_size)                        # batch * att_size
        weight = F.softmax(dot, dim=1)                             # batch * att_size
        if att_masks is not None:
            weight = weight * att_masks.view(-1, att_size).float()
            weight = weight / weight.sum(1, keepdim=True) # normalize to 1
        att_feats_ = att_feats.contiguous().view(-1, att_size, att_feats.size(-1)).contiguous() # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1) # batch * att_feat_size

        return att_res



class densepembedTopDownModel(densepembedAttModel):
    def __init__(self, opt):
        super(densepembedTopDownModel, self).__init__(opt)
        self.num_layers = 2
        self.core = TopDownCore(opt)

