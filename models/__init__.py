from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy

import numpy as np
import misc.utils as utils
import torch


from .densepembedAttModel  import *
__all__ = ['setup', 'load', 'JointModel']

def setup(opt,caption=True):
    if caption:
        # Top-down attention model
        if opt.caption_model == 'topdown':
            model = TopDownModel(opt)
        # StackAtt
        elif opt.caption_model == 'topdown_pembed':
            model = TopDownPembedModel(opt)
        elif opt.caption_model == 'densepembed':
            model = densepembedTopDownModel(opt)
        elif opt.caption_model == "densemix_pembed_noperson":
            model = densemix_pembedTopDownModel_nop(opt)
        else:
            raise Exception("caption model not supported: {}".format(opt.caption_model))
    else:
        raise Exception("model not supported for emptry caption")
    # check compatibility if training is continued from previously saved model
    if vars(opt).get('start_from', None) is not None:
        # check if all necessary files exist 
        assert os.path.isdir(opt.start_from)," %s must be a a path" % opt.start_from
        assert os.path.isfile(os.path.join(opt.start_from,"infos_"+opt.id+".pkl")),"infos.pkl file does not exist in path %s"%opt.start_from
        pretrained_dict = torch.load(os.path.join(opt.start_from, 'model.pth'))
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    return model
