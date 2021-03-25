import torch
import misc.utils as utils
from misc.rewards import init_scorer, get_self_critical_reward
def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()
class LossWrapper(torch.nn.Module):
    def __init__(self, model, opt):
        super(LossWrapper, self).__init__()
        self.opt = opt
        self.model = model
        if opt.label_smoothing > 0:
            self.crit = utils.LabelSmoothing(smoothing=opt.label_smoothing)
        else:
            self.crit = utils.LanguageModelCriterion()
    
    def entropy(self,input, seq):
        input = to_contiguous(input)
        mask_en = (seq>0).float()
        mask_en = to_contiguous(torch.cat([mask_en.new(mask_en.size(0), 1).fill_(1), mask_en[:, :-1]], 1))
        output = - input* mask_en
        output = torch.sum(output) / torch.sum(mask_en)

        return output

    def forward(self, fc_feats, att_feats,densecap, labels, masks, att_masks,personality, gts, gt_indices,
                sc_flag):
        out = {}
        if not sc_flag:
            loss = self.crit(self.model(fc_feats, att_feats,densecap, labels, att_masks,personality),labels[:,1:], masks[:,1:])
        out['loss'] = loss
        return out
