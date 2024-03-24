"""
Run evaluation with saved models.
"""

import os
import random
import argparse
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from data.loader import DataLoader
from model.cnn import RelationModel
from utils import torch_utils, scorer, constant, helper
from utils.vocab import Vocab

parser = argparse.ArgumentParser()
parser.add_argument('model_dir', type=str, help='Directory of the model.')
parser.add_argument('--model', type=str, default='best_model.pt', help='Name of the model file.')
parser.add_argument('--data_dir', type=str, default='dataset/tacred')
parser.add_argument('--dataset', type=str, default='test', help="Evaluate on dev or test.")


#Analysis & Logging
parser.add_argument('--out', type=str, default='', help="Save model predictions to this dir.")
parser.add_argument('--prediction_logs', action='store_true', help="Generate prediction logs.")
parser.add_argument('--dump_cm', action='store_true', help="Dump Confusion Matrix into a file. \
						Must be followed by --out <log_files_dir> for dumping confusion matrix pickle.")
parser.add_argument('--topk', action='store_true', help='Consider top-k Evaluation. Must be followed by --k <integer>.')
parser.add_argument('--k', type=int, default=1, help='Value of k for top-k Evaluation.')
parser.add_argument('--tsne', action='store_true', help='Generate tSNE plots.')
parser.add_argument('--batch_size', type=int, default=0, help='Sets batch size to generate tSNE plot')

#Dataset Manipulation
parser.add_argument('--binary', action='store_true', help="Train a Binary Classifier")
parser.add_argument('--downsample_num', type=int, default=1,
                    help="No relation downsampling numberator. Make 0 to remove all no_relations")
parser.add_argument('--downsample_den', type=int, default=1, help="No relation downsampling denominator")

parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true')

#Budget Reannotation
parser.add_argument('--align_retacred', action='store_true', 
                        help="Fetches Dataset eqivalent to unambiguous retacred")
args = parser.parse_args()

if args.binary:
    constant.LABEL_TO_ID = constant.BINARY_LABEL_TO_ID

torch.manual_seed(args.seed)
random.seed(1234)
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)

# load opt
model_file = args.model_dir + '/' + args.model
print("Loading model from {}".format(model_file))
opt = torch_utils.load_config(model_file)
model = RelationModel(opt)
model.load(model_file)

# Appending Evaluation Analysis arguements to Model parameters
opt['out'] = args.out
opt['model_dir'] = args.model_dir
opt['dataset'] = args.dataset
opt['prediction-logs'] = args.prediction_logs
opt['model_name'] = args.model_dir.split('/')[-1]
opt['dump_cm'] = args.dump_cm
opt['topk'] = args.topk
opt['binary'] = args.binary
# @Akshay Setting up batch size for tSNE visulatization
if args.batch_size:
	print("BATCH SIZE  ::  {}".format(args.batch_size))
	opt['batch_size'] = args.batch_size

# load vocab
vocab_file = args.model_dir + '/vocab.pkl'
vocab = Vocab(vocab_file, load=True)
assert opt['vocab_size'] == vocab.size, "Vocab size must match that in the saved model."

# load data
data_file = opt['data_dir'] + '/{}.json'.format(args.dataset)
print("Loading data from {} with batch size {}...".format(data_file, opt['batch_size']))
batch = DataLoader(data_file, opt['batch_size'], opt, vocab, evaluation=True,
                   downsample=[args.downsample_num, args.downsample_den])

helper.print_config(opt)
id2label = dict([(v,k) for k,v in constant.LABEL_TO_ID.items()])

predictions = []
all_probs = []
for i, b in enumerate(batch):
    preds, probs, _ = model.predict(b, tsne=args.tsne, step=i)
    if args.topk:
    	preds = np.argsort(probs, axis=1)[:, -1 * args.k:].tolist()
    predictions += preds
    all_probs += np.max(probs, axis=1).tolist()
if args.topk:
	predictions = [[id2label[p] for p in prediction] for prediction in predictions]
else:
	predictions = [id2label[p] for p in predictions]
p, r, f1 = scorer.score(batch.sentence_ids(), batch.gold(), predictions, all_probs, verbose=True, opt=opt)
print("{} set evaluate result: {:.2f}\t{:.2f}\t{:.2f}".format(args.dataset,p,r,f1))

print("Evaluation ended.")

