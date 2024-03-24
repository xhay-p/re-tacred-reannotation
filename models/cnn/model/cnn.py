"""
A cnn model for relation extraction, written in pytorch.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F

from utils import constant, torch_utils
from model.loss_functions import HierarchicalDistanceLoss
from visualisation import tSNE
# from model import layers

class RelationModel(object):
    """ A wrapper class for the training and evaluation of models. """
    def __init__(self, opt, emb_matrix=None):
        self.opt = opt
        self.model = CNN(opt, emb_matrix)
        if 'hier_dist' in opt and opt['hier_dist']:
            self.criterion = HierarchicalDistanceLoss(opt)
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if opt['cuda']:
            self.model.cuda()
            self.criterion.cuda()
        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.parameters, opt['lr'], 
                                                    weight_decay=opt['weight_decay'])
    
    def update(self, batch):
        """ Run a step of forward and backward model update. """
        if self.opt['cuda']:
            inputs = [b.cuda() for b in batch[:7]]
            labels = batch[7].cuda()
        else:
            inputs = [b for b in batch[:7]]
            labels = batch[7]

        # step forward
        self.model.train()
        self.optimizer.zero_grad()
        logits, _ = self.model(inputs)
        if self.opt['hier_dist']:
            loss, distance_factor = self.criterion(logits, labels)
        else:
            loss = self.criterion(logits, labels)
        
        # backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt['max_grad_norm'])
        self.optimizer.step()
        loss_val = loss.data.item()
        return loss_val

    def predict(self, batch, unsort=True, tsne=False, step=0):
        """ Run forward prediction. If unsort is True, recover the original order of the batch. """
        if self.opt['cuda']:
            inputs = [b.cuda() for b in batch[:7]]
            labels = batch[7].cuda()
        else:
            inputs = [b for b in batch[:7]]
            labels = batch[7]

        orig_idx = batch[8]

        # forward
        self.model.eval()
        logits, final_hidden = self.model(inputs)
        if tsne:
            tSNE.tsne_plot(logits, labels, step)
        if 'hier_dist' in self.opt and self.opt['hier_dist']:
            loss, _ = self.criterion(logits, labels)
        else:
            loss = self.criterion(logits, labels)
        probs = F.softmax(logits, dim=1).data.cpu().numpy().tolist()
        predictions = np.argmax(logits.data.cpu().numpy(), axis=1).tolist()
        if unsort:
            _, predictions, probs = [list(t) for t in zip(*sorted(zip(orig_idx,\
                    predictions, probs)))]
        # return predictions, probs, loss.data.item(), final_hidden.detach().cpu().numpy().tolist()
        return predictions, probs, loss.data.item()

    def update_lr(self, new_lr):
        torch_utils.change_lr(self.optimizer, new_lr)

    def save(self, filename, epoch):
        params = {
                'model': self.model.state_dict(),
                'config': self.opt,
                'epoch': epoch
                }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.opt = checkpoint['config']

class CNN(nn.Module):
    """ A sequence model for relation extraction. """

    def __init__(self, opt, emb_matrix=None):
        super(CNN, self).__init__()
        self.drop = nn.Dropout(opt['dropout'])
        self.emb = nn.Embedding(opt['vocab_size'], opt['emb_dim'], padding_idx=constant.PAD_ID)
        if opt['pos_dim'] > 0:
            self.pos_emb = nn.Embedding(len(constant.POS_TO_ID), opt['pos_dim'],
                    padding_idx=constant.PAD_ID)
        if opt['ner_dim'] > 0:
            self.ner_emb = nn.Embedding(len(constant.NER_TO_ID), opt['ner_dim'],
                    padding_idx=constant.PAD_ID)
        
        self.pe_emb = nn.Embedding(constant.MAX_LEN * 2 + 3, opt['pe_dim'])

        input_size = opt['emb_dim'] + opt['pos_dim'] + opt['ner_dim'] + opt['pe_dim'] * 2

        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1,
                                              out_channels=opt['filter_num'],
                                              kernel_size=(k, input_size),
                                              padding=0) for k in opt['filters']])
        self.linear = nn.Linear(opt['filter_num'] * len(opt['filters']), opt['num_class'] )

        self.opt = opt
        self.topn = self.opt.get('topn', 1e10)
        self.use_cuda = opt['cuda']
        self.emb_matrix = emb_matrix
        self.init_weights()
    
    def init_weights(self):
        if self.emb_matrix is None:
            self.emb.weight.data[1:,:].uniform_(-1.0, 1.0) # keep padding dimension to be 0
        else:
            self.emb_matrix = torch.from_numpy(self.emb_matrix)
            self.emb.weight.data.copy_(self.emb_matrix)
        if self.opt['pos_dim'] > 0:
            self.pos_emb.weight.data[1:,:].uniform_(-1.0, 1.0)
        if self.opt['ner_dim'] > 0:
            self.ner_emb.weight.data[1:,:].uniform_(-1.0, 1.0)

        self.linear.bias.data.fill_(0)
        init.xavier_uniform_(self.linear.weight, gain=1) # initialize linear layer
        
        # decide finetuning
        if self.topn <= 0:
            print("Do not finetune word embedding layer.")
            self.emb.weight.requires_grad = False
        elif self.topn < self.opt['vocab_size']:
            print("Finetune top {} word embeddings.".format(self.topn))
            self.emb.weight.register_hook(lambda x: \
                    torch_utils.keep_partial_grad(x, self.topn))
        else:
            print("Finetune all embeddings.")
   
    def forward(self, inputs):
        words, masks, pos, ner, deprel, subj_pos, obj_pos = inputs # unpack
        seq_lens = list(masks.data.eq(constant.PAD_ID).long().sum(1).squeeze())
        batch_size = words.size()[0]
        
        # embedding lookup
        word_inputs = self.emb(words)
        inputs = [word_inputs]
        if self.opt['pos_dim'] > 0:
            inputs += [self.pos_emb(pos)]
        if self.opt['ner_dim'] > 0:
            inputs += [self.ner_emb(ner)]
        if self.opt['pe_dim']:
            # convert all negative PE numbers to positive indices
            # e.g., -2 -1 0 1 will be mapped to 98 99 100 101
            inputs += [self.pe_emb(subj_pos + constant.MAX_LEN)]
            inputs += [self.pe_emb(obj_pos + constant.MAX_LEN)]
        # batch_size x batch_max_len x feature_dim
        inputs = self.drop(torch.cat(inputs, dim=2).unsqueeze(1)) # add dropout to input
        # batch_size x 1 x batch_max_len x feature_dim
        input_size = inputs.size(2)
        
        #cnn
        hidden = [torch.tanh(conv(inputs)).squeeze(3) for conv in self.convs]
        hidden = [F.max_pool1d(i, kernel_size=i.size(2)).squeeze(2) for i in hidden]
        sentence_features = torch.cat(hidden, dim=1)
        hidden = self.drop(sentence_features)
        final_hidden = hidden
        # # attention
        # if self.opt['attn']:
            # convert all negative PE numbers to positive indices
            # e.g., -2 -1 0 1 will be mapped to 98 99 100 101
        #     subj_pe_inputs = self.pe_emb(subj_pos + constant.MAX_LEN)
        #     obj_pe_inputs = self.pe_emb(obj_pos + constant.MAX_LEN)
        #     pe_features = torch.cat((subj_pe_inputs, obj_pe_inputs), dim=2)
        #     final_hidden = self.attn_layer(outputs, masks, hidden, pe_features)
        # else:
        #     final_hidden = hidden

        logits = self.linear(final_hidden)
        return logits, final_hidden
    

