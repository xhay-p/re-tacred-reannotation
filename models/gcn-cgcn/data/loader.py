"""
Data loader for TACRED json files.
"""
import os
import json
import random
import torch
import numpy as np

from utils import constant, helper, vocab

class DataLoader(object):
    """
    Load data from json files, preprocess and prepare batches.
    """
    def __init__(self, filename, batch_size, opt, vocab, evaluation=False, downsample=[1, 1]):
        self.batch_size = batch_size
        self.opt = opt
        self.vocab = vocab
        self.eval = evaluation
        self.label2id = constant.LABEL_TO_ID

        with open(filename) as infile:
            data = json.load(infile)
        self.raw_data = data

        dataset = filename.split('/')[-1].split('.')[0]

        if 'align_retacred' in self.opt and self.opt['align_retacred']:
            patch_path = os.path.join(self.opt['data_dir'], 'Re-TACRED/' + dataset + '_id2label.json')
            self.valid_sids = json.load(open(patch_path, 'r'))
            print("Number of Valid sids ::  {}".format(len(self.valid_sids)))

        data = self.preprocess(data, vocab, opt, downsample, dataset)

        # shuffle for training
        if not evaluation:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
        self.id2label = dict([(v,k) for k,v in self.label2id.items()])
        self.labels = [self.id2label[d[-2]] for d in data]
        self.sids = [d[-1] for d in data]
        self.num_examples = len(data)
        print("{} examples in {}".format(self.num_examples, filename))

        # chunk into batches
        data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        self.data = data
        print("{} batches created for {}".format(len(data), filename))

    def preprocess(self, data, vocab, opt, downsample, dataset):
        """ Preprocess the data and convert to ids. """
        if 'repalce' in opt and opt['replace'] and dataset == 'train':
            replace_ids = pickle.load(open(opt['replace'], 'rb'))
            print("Number of labels to change from negative to positive :: {}".format(len(replace_ids)))
        processed = []
        index = 0
        for d in data:
            # @Akshay Adding sentence id to identify incorrect sentences
            s_id = d['id']
            if s_id not in self.valid_sids:
                continue
            tokens = list(d['token'])
            if opt['lower']:
                tokens = [t.lower() for t in tokens]
            # anonymize tokens
            ss, se = d['subj_start'], d['subj_end']
            os, oe = d['obj_start'], d['obj_end']
            tokens[ss:se+1] = ['SUBJ-'+d['subj_type']] * (se-ss+1)
            tokens[os:oe+1] = ['OBJ-'+d['obj_type']] * (oe-os+1)
            tokens = map_to_ids(tokens, vocab.word2id)
            pos = map_to_ids(d['stanford_pos'], constant.POS_TO_ID)
            ner = map_to_ids(d['stanford_ner'], constant.NER_TO_ID)
            deprel = map_to_ids(d['stanford_deprel'], constant.DEPREL_TO_ID)
            head = [int(x) for x in d['stanford_head']]
            assert any([x == 0 for x in head])
            l = len(tokens)
            subj_positions = get_positions(d['subj_start'], d['subj_end'], l)
            obj_positions = get_positions(d['obj_start'], d['obj_end'], l)
            subj_type = [constant.SUBJ_NER_TO_ID[d['subj_type']]]
            obj_type = [constant.OBJ_NER_TO_ID[d['obj_type']]]
            # Binary Classification: Mapping positive relations to label 'relation'
            if opt['binary'] and d['relation'] in constant.BINARY_MAP:
                relation = self.label2id[constant.BINARY_MAP[d['relation']]]
            elif 'repalce' in opt and opt['replace'] and dataset == 'train' and s_id in replace_ids:
                relation = constant.LABEL_TO_ID[replace_ids[s_id]]
            else:
                relation = self.label2id[d['relation']]
            # Applying DownSampling
            if relation != 0:
                processed += [(tokens, pos, ner, deprel, head, subj_positions, obj_positions, subj_type, obj_type, relation, s_id)]
            else:
                if index < downsample[0]:
                    processed += [(tokens, pos, ner, deprel, head, subj_positions, obj_positions, subj_type, obj_type, relation, s_id)]
        return processed

    def gold(self):
        """ Return gold labels as a list. """
        return self.labels

    def sentence_ids(self):
        """ Return Sentence ids as a list. """
        return self.sids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch_size = len(batch)
        batch = list(zip(*batch))
        assert len(batch) == 11

        # sort all fields by lens for easy RNN operations
        lens = [len(x) for x in batch[0]]
        batch, orig_idx = sort_all(batch, lens)

        # word dropout
        if not self.eval:
            words = [word_dropout(sent, self.opt['word_dropout']) for sent in batch[0]]
        else:
            words = batch[0]

        # convert to tensors
        words = get_long_tensor(words, batch_size)
        masks = torch.eq(words, 0)
        pos = get_long_tensor(batch[1], batch_size)
        ner = get_long_tensor(batch[2], batch_size)
        deprel = get_long_tensor(batch[3], batch_size)
        head = get_long_tensor(batch[4], batch_size)
        subj_positions = get_long_tensor(batch[5], batch_size)
        obj_positions = get_long_tensor(batch[6], batch_size)
        subj_type = get_long_tensor(batch[7], batch_size)
        obj_type = get_long_tensor(batch[8], batch_size)

        rels = torch.LongTensor(batch[9])

        return (words, masks, pos, ner, deprel, head, subj_positions, obj_positions, subj_type, obj_type, rels, orig_idx)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

def map_to_ids(tokens, vocab):
    ids = [vocab[t] if t in vocab else constant.UNK_ID for t in tokens]
    return ids

def get_positions(start_idx, end_idx, length):
    """ Get subj/obj position sequence. """
    return list(range(-start_idx, 0)) + [0]*(end_idx - start_idx + 1) + \
            list(range(1, length-end_idx))

def get_long_tensor(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded LongTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.LongTensor(batch_size, token_len).fill_(constant.PAD_ID)
    for i, s in enumerate(tokens_list):
        tokens[i, :len(s)] = torch.LongTensor(s)
    return tokens

def sort_all(batch, lens):
    """ Sort all fields by descending order of lens, and return the original indices. """
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    return sorted_all[2:], sorted_all[1]

def word_dropout(tokens, dropout):
    """ Randomly dropout tokens (IDs) and replace them with <UNK> tokens. """
    return [constant.UNK_ID if x != constant.UNK_ID and np.random.random() < dropout \
            else x for x in tokens]

