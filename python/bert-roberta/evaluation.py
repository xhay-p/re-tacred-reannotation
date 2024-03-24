import numpy as np
from hierarchy_utils import constant, hierarchy

d, _ = hierarchy.get_dis_lca_matrix()
DIS_MATRIX = np.array(d)
DIS_MAX = np.max(DIS_MATRIX)

def get_f1(key, prediction):
    correct_by_relation = ((key == prediction) & (prediction != 0)).astype(np.int32).sum()
    guessed_by_relation = (prediction != 0).astype(np.int32).sum()
    gold_by_relation = (key != 0).astype(np.int32).sum()

    prec_micro = 1.0
    if guessed_by_relation > 0:
        prec_micro = float(correct_by_relation) / float(guessed_by_relation)
    recall_micro = 1.0
    if gold_by_relation > 0:
        recall_micro = float(correct_by_relation) / float(gold_by_relation)
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
    return prec_micro, recall_micro, f1_micro

def compute_f1(predictions, labels):
    n_gold = n_pred = n_correct = 0
    for pred, label in zip(predictions, labels):
        if pred != 0:
            n_pred += 1
        if label != 0:
            n_gold += 1
        if (pred != 0) and (label != 0) and (pred == label):
            n_correct += 1
    if n_correct == 0:
        prec = recall = f1 = 0.0
    else:
        prec = n_correct * 1.0 / n_pred
        recall = n_correct * 1.0 / n_gold
        if prec + recall > 0:
            f1 = 2.0 * prec * recall / (prec + recall)
        else:
            f1 = 0.0
    print("SpanBERT Evaluation")
    print("precision :: {} | Recall  ::  {}  |  F1  ::  {}\n".format(prec, recall, f1))

def compute_h_f1(predictions, labels):
    G = hierarchy.generate_graph()
    n_gold = n_pred = n_correct = 0
    for pred, label in zip(predictions, labels):
        if pred != 0:
            n_pred += 1
        if label != 0:
            n_gold += 1
        if (pred != 0) and (label != 0) and (pred == label):
            n_correct += 1
        if (pred != 0) and (label != 0) and (pred != label):
            s, _ = hierarchy.common_path_score(G, label, pred)
            n_correct += s
    if n_correct == 0:
        prec = recall = f1 = 0.0
    else:
        prec = n_correct * 1.0 / n_pred
        recall = n_correct * 1.0 / n_gold
        if prec + recall > 0:
            f1 = 2.0 * prec * recall / (prec + recall)
        else:
            f1 = 0.0
    print("Hierarchical Evaluation")
    print("precision :: {} | Recall  ::  {}  |  F1  ::  {}\n".format(prec, recall, f1))

def accuracy(preds, labels):
    print("Accuracy :: {:.3%}\n".format((preds == labels).mean()))

def hierarchical_accuracy(preds, labels):
    score = 0
    count = 0
    n_count = 0
    pred_dist = {}
    # load hierarchy
    G = hierarchy.generate_graph()
    for label, pred in zip(labels, preds):
        if label == 0 and pred == 0:
            n_count += 1

        elif label != 0:
            if pred == 0:
                s, dist = 0, DIS_MAX * 2
            else:
                s, dist = hierarchy.common_path_score(G, label, pred)
            score += s
            pred_dist[dist] = pred_dist.get(dist,0) + 1
            count += 1

    h_accuracy = score/count

    print("Hierarchical Positive Accuracy: {:.3%}".format(h_accuracy))
    print("Predictions @ distances: ", pred_dist)

def positive_accuracy(preds, labels):
    count = 0
    n_count = 0
    for label, pred in zip(labels, preds):
        if label == pred and label != 0:
            count += 1
        if label == 0:
            n_count += 1
    accuracy = count/(len(labels) - n_count)
    print("Positive Accuracy: {} out of {} predictions are correct. Accuracy : {:.3%}".format(count, len(labels) - n_count, accuracy))