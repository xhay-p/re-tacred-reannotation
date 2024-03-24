import json
from utils import constant
import random


class ResultAnalysis(object):
    def __init__(self, opt, data_file, output_file):
        self.opt = opt
        with open(data_file) as infile:
            self.data = json.load(infile)
        self.f_results = open(output_file, "w")
        self.subj_symbol = '#'
        self.obj_symbol = '$'
        self.results = []
        self.id2label = dict([(v, k) for k, v in constant.LABEL_TO_ID.items()])

    def __del__(self):
        self.f_results.close()

    def analyse_errors(self, preds, probs, batch_cnt, batch_size):
        start_ind = batch_cnt * self.opt['batch_size']
        end_ind = start_ind + batch_size
        for i in range(start_ind, end_ind):
            j = i-start_ind
            relation = self.data[i]['relation']
            if preds[j] == constant.LABEL_TO_ID[relation]:
                continue

            id = self.data[i]['id']
            tokens = self.data[i]['token']
            ss, se = self.data[i]['subj_start'], self.data[i]['subj_end']
            os, oe = self.data[i]['obj_start'], self.data[i]['obj_end']

            tokens[ss] = self.subj_symbol + tokens[ss]
            tokens[se] = tokens[se] + self.subj_symbol
            tokens[os] = self.obj_symbol + tokens[os]
            tokens[oe] = tokens[oe] + self.obj_symbol

            probs[j] = [round(p, 4) for p in probs[j]]
            probs[j] = {k: v for k, v in enumerate(probs[j])}

            result = {'id': id,
                      'relation': relation,
                      'prediction': self.id2label[preds[j]],
                      'sentence': ' '.join(tokens),
                      'logits': probs[j]
                      }

            self.results.append(result)

    def print_results(self):
        random.shuffle(self.results)
        self.f_results.write(json.dumps(self.results))