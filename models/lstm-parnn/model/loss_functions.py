import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import pickle

from utils import hierarchy, constant

class HierarchicalDistanceLoss(torch.nn.Module):

    def __init__(self, opt):
        super(HierarchicalDistanceLoss, self).__init__()
        self.cross_loss_function = nn.CrossEntropyLoss(reduction="none")
        d, l = hierarchy.get_dis_lca_matrix()
        self.DIS_MATRIX = np.array(d)
        self.LCA_MATRIX  = np.array(l)
        self.normalise = opt['hier_normalise']
        self.opt = opt
        self.id2label = {val:key for key, val in constant.LABEL_TO_ID.items()}
        print(self.id2label)
        self.DIS_MAX = np.max(self.DIS_MATRIX)
        print("Maximum graph distance :: {}".format(self.DIS_MAX))

    def get_hierarchical_dist(self, pred, gold):
        if self.id2label[gold] == 'no_relation' and self.id2label[pred] != 'no_relation':
            return self.DIS_MAX * 2
        elif self.id2label[pred] == 'no_relation' and self.id2label[gold] != 'no_relation':
            return self.DIS_MAX
        else:
            return self.DIS_MATRIX[gold, pred]

    def get_pred_lca_dist(self, pred, gold):
        if self.id2label[gold] == 'no_relation' and self.id2label[pred] != 'no_relation':
            return self.DIS_MAX * 2
        elif self.id2label[pred] == 'no_relation' and self.id2label[gold] != 'no_relation':
            return self.DIS_MAX
        else:
            return self.DIS_MATRIX[gold, self.LCA_MATRIX[gold, pred]]

    def forward(self, logits, labels):
        ce_loss_all = self.cross_loss_function(logits, labels)
        predictions = np.argmax(logits.data.cpu().numpy(), axis=1).tolist()
        target = labels.cpu().numpy()
        if not self.opt['lca']:
            distance_factor = torch.from_numpy(np.array(list(map(self.get_hierarchical_dist, predictions, target))))
        else:
            distance_factor = torch.from_numpy(np.array(list(map(self.get_pred_lca_dist, predictions, target))))
        distance_factor = distance_factor.to(ce_loss_all.device)
        ce_loss_all = torch.mul(ce_loss_all, distance_factor)
        total_loss = ce_loss_all.mean() /self.normalise
        return total_loss, distance_factor

