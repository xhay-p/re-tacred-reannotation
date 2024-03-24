#!/usr/bin/env python

"""
Score the predictions with gold labels, using precision, recall and F1 metrics.
"""

import argparse
import sys
from collections import Counter
import networkx as nx
from prettytable import PrettyTable
import pickle

from utils import hierarchy, constant, helper

NO_RELATION = "no_relation"

def parse_arguments():
    parser = argparse.ArgumentParser(description='Score a prediction file using the gold labels.')
    parser.add_argument('gold_file', help='The gold relation file; one relation per line')
    parser.add_argument('pred_file', help='A prediction file; one relation per line, in the same order as the gold file.')
    args = parser.parse_args()
    return args

def generate_per_relation_statistics(correct_by_relation, guessed_by_relation, gold_by_relation):
    print("#####Per-Relation Statistics:#####")
    relations = gold_by_relation.keys()
    longest_relation = 0
    for relation in sorted(relations):
        longest_relation = max(len(relation), longest_relation)
    for relation in sorted(relations):
        # (compute the score)
        correct = correct_by_relation[relation]
        guessed = guessed_by_relation[relation]
        gold    = gold_by_relation[relation]
        prec = 1.0
        if guessed > 0:
            prec = float(correct) / float(guessed)
        recall = 0.0
        if gold > 0:
            recall = float(correct) / float(gold)
        f1 = 0.0
        if prec + recall > 0:
            f1 = 2.0 * prec * recall / (prec + recall)
        # (print the score)
        sys.stdout.write(("{:<" + str(longest_relation) + "}").format(relation))
        sys.stdout.write("  P: ")
        if prec < 0.1: sys.stdout.write(' ')
        if prec < 1.0: sys.stdout.write(' ')
        sys.stdout.write("{:.2%}".format(prec))
        sys.stdout.write("  R: ")
        if recall < 0.1: sys.stdout.write(' ')
        if recall < 1.0: sys.stdout.write(' ')
        sys.stdout.write("{:.2%}".format(recall))
        sys.stdout.write("  F1: ")
        if f1 < 0.1: sys.stdout.write(' ')
        if f1 < 1.0: sys.stdout.write(' ')
        sys.stdout.write("{:.2%}".format(f1))
        sys.stdout.write("  #: %5d" % gold)
        sys.stdout.write("  c: %5d" % correct)
        sys.stdout.write("\n")
    print("")

def calculate_micro_prf1(correct_by_relation, guessed_by_relation, gold_by_relation):
    prec_micro = 1.0
    if sum(guessed_by_relation.values()) > 0:
        prec_micro   = float(sum(correct_by_relation.values())) / float(sum(guessed_by_relation.values()))
    recall_micro = 0.0
    if sum(gold_by_relation.values()) > 0:
        recall_micro = float(sum(correct_by_relation.values())) / float(sum(gold_by_relation.values()))
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)


    print(
        "Calculating Precision:",
        "correct_by_relation=",
        float(sum(correct_by_relation.values())),
        "/",
        "guessed_by_relation=",
        float(sum(guessed_by_relation.values()))
    )
    print(
        "Calculating Recall:",
        "correct_by_relation=",
        float(sum(correct_by_relation.values())),
        "/",
        "gold_by_relation=",
        float(sum(gold_by_relation.values()))
    )

    return prec_micro, recall_micro, f1_micro

def score(sids, key, prediction, confidence, verbose=False, opt=None):
    correct_by_relation = Counter()
    guessed_by_relation = Counter()
    gold_by_relation    = Counter()

    # To keep track of positive and negative accuracy
    no_rel = crct_no_rel = rel = crct_rel = 0

    if opt:
        topk = opt['topk']
        logging = opt['prediction_logs']
        dump_cm = opt['dump_cm']
        if logging:
            G = hierarchy.generate_graph()
            cc = inc = 0
            out = opt['out']
            if out:
                helper.ensure_dir(out)
                model_dir = opt['model_dir']
                dataset = opt['dataset']
                incorrects_file = out + '/inc_{}_{}.tsv'.format(model_dir.split('/')[-1], dataset)
                incorrect_logger = helper.FileLogger(incorrects_file, header="sentence_id\tground_truth\tprediction\tconfidence\tdp\tdl\tdr\tdpr\tdlr\tdlp")
                corrects_file = out + '/c_{}_{}.tsv'.format(model_dir.split('/')[-1], dataset)
                correct_logger = helper.FileLogger(corrects_file, header="sentence_id\tground_truth\tprediction\tconfidence\tdp\tdl\tdr\tdpr\tdlr\tdlp")
            else:
                incorrect_logger = correct_logger = None
    else:
        logging = False
        topk = False
        dump_cm = False

    #Confusion Matrix
    num_class = len(constant.LABEL_TO_ID)
    confusion_matrix = [[0 for x in range(num_class)] for y in range(num_class)]

    # Loop over the data to compute a score
    for row in range(len(key)):
        sid = sids[row]
        gold = key[row]
        if topk:
            guess = prediction[row][-1]
        else:
            guess = prediction[row]
        conf = confidence[row]
        gold_id = constant.LABEL_TO_ID[gold]
        guess_id = constant.LABEL_TO_ID[guess]

        # Updating Confusion Matrix
        confusion_matrix[gold_id][guess_id] += 1
         
        if gold == NO_RELATION and guess == NO_RELATION:
            no_rel += 1
            crct_no_rel += 1
            pass
        elif gold == NO_RELATION and guess != NO_RELATION:
            no_rel += 1
            guessed_by_relation[guess] += 1
        elif gold != NO_RELATION and guess == NO_RELATION:
            rel += 1
            gold_by_relation[gold] += 1
        elif gold != NO_RELATION and guess != NO_RELATION:
            rel += 1
            guessed_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                crct_rel += 1
                correct_by_relation[guess] += 1

        # Generating prediction logs
        if logging:
            # distance of prediction from label
            dp = nx.shortest_path_length(G, gold_id, guess_id)
            lca = hierarchy.get_lca(G, gold_id, guess_id)
            # distance of LCA from label
            dl = nx.shortest_path_length(G, gold_id, lca)
            # distance of root from label
            dr = nx.shortest_path_length(G, gold_id, 59)
            # distance of prediction from root
            dpr = nx.shortest_path_length(G, guess_id, 59)
            # distance of root from lca
            dlr = nx.shortest_path_length(G, lca, 59)
            # distance of prediction from lca
            dlp = nx.shortest_path_length(G, guess_id, lca)
            
            if gold_id != guess_id:
                if incorrect_logger:
                    incorrect_logger.log("{}\t{}\t{}\t{:.3f}\t{}\t{}\t{}\t{}\t{}\t{}".format(sid, gold_id, guess_id, conf, dp, dl, dr, dpr, dlr, dlp))
                inc += 1
            else:
                if correct_logger:
                    correct_logger.log("{}\t{}\t{}\t{:.3f}\t{}\t{}\t{}\t{}\t{}\t{}".format(sid, gold_id, guess_id, conf, dp, dl, dr, dpr, dlr, dlp))
                cc += 1
            print("{}\t{}\t{}\t{}\t{:.3f}\t{}\t{}\t{}\t{}\t{}\t{}".format((row+1), sid, gold, guess, conf, dp, dl, dr, dpr, dlr, dlp))
    print()
    if logging:
        print("Total Incorrect Predictions :: {}".format(inc))
        print("Total Correct Predictions :: {}".format(cc))

    # Print verbose information
    if verbose:
        generate_per_relation_statistics(correct_by_relation, guessed_by_relation, gold_by_relation)

    # Print the aggregate score
    if verbose:
        print("Final Score:")
    prec_micro, recall_micro, f1_micro = calculate_micro_prf1(correct_by_relation, guessed_by_relation, gold_by_relation)
    print( "Precision (micro): {:.3%}".format(prec_micro) )
    print( "   Recall (micro): {:.3%}".format(recall_micro) )
    print( "       F1 (micro): {:.3%}".format(f1_micro) )
    
    # Print the Accuracy
    if no_rel != 0 and rel != 0:
        print(" Negative Accuracy : {} out of {}: {:.3%}".format(crct_no_rel, no_rel, (crct_no_rel / no_rel)))
        print(" Positive Accuracy : {} out of {}: {:.3%}".format(crct_rel, rel, (crct_rel / rel)))
        print(" Overall Accuracy : {} out of {}: {:.3%}".format((crct_rel+crct_no_rel), (rel+no_rel), 
                                                                    ((crct_rel+crct_no_rel) / (rel+no_rel))))
    print()

    # top-k Predictions Results Analysis
    if topk:
        print("##### Top-k Evaluation #####")
        correct_by_relation = Counter()
        guessed_by_relation = Counter()
        gold_by_relation    = Counter()
        correct_distribution = [0] * (len(prediction[0]) + 1)
        for row in range(len(key)):
            gold = key[row]
            guess = prediction[row][-1]
            all_guesses = prediction[row]
            gold_id = constant.LABEL_TO_ID[gold]
            guess_id = constant.LABEL_TO_ID[guess]

            if gold in all_guesses:
                correct_distribution[all_guesses.index(gold)] += 1
                if gold == NO_RELATION:
                    pass
                else:
                    guessed_by_relation[gold] += 1
                    gold_by_relation[gold] += 1
                    correct_by_relation[gold] += 1
            else:
                correct_distribution[-1] += 1
                if gold == NO_RELATION:
                    guessed_by_relation[guess] += 1
                elif guess == NO_RELATION:
                    gold_by_relation[gold] += 1
                else:
                    guessed_by_relation[guess] += 1
                    gold_by_relation[gold] += 1
        if verbose:
            generate_per_relation_statistics(correct_by_relation, guessed_by_relation, gold_by_relation)
        
        if verbose:
            print("Final Score:")
        prec_micro_tk, recall_micro_tk, f1_micro_tk = calculate_micro_prf1(correct_by_relation, guessed_by_relation, gold_by_relation)
        
        print("Precision (micro): {:.3%}".format(prec_micro_tk))
        print("   Recall (micro): {:.3%}".format(recall_micro_tk))
        print("       F1 (micro): {:.3%}".format(f1_micro_tk))
        print("Top K guesses")
        total_guesses = sum(correct_distribution)
        correct_distribution = [i / total_guesses for i in correct_distribution]
        top_k = len(correct_distribution)
        for i in range(top_k - 1):
            print("Top {} : {:.3%}".format(top_k - i - 1, correct_distribution[i]))
        print("Wrong guesses : {:.3%}".format(correct_distribution[-1]))
        print("\n")    

    # Printing Confusion Matrix
    if verbose:
        id2label = dict([(v, k) for k, v in constant.LABEL_TO_ID.items()])
        class_id = [x for x in range(num_class)]
        class_id.insert(0, "")
        p = PrettyTable()
        p.add_row(class_id)
        cnt = 0
        for row in confusion_matrix:
            row.insert(0, id2label[cnt])
            p.add_row(row)
            cnt += 1

        # print("##### Confusion Matrix #####")
        # p.align = 'l'
        # print(p.get_string(header=False, border=False))
        # print("")
        if dump_cm:
            out_dir = opt['out']+'/confusion_matrix/'
            helper.ensure_dir(out_dir)
            pickle.dump(confusion_matrix, open(out_dir+opt['model_name'], 'wb'))

    return prec_micro, recall_micro, f1_micro

if __name__ == "__main__":
    # Parse the arguments from stdin
    args = parse_arguments()
    key = [str(line).rstrip('\n') for line in open(str(args.gold_file))]
    prediction = [str(line).rstrip('\n') for line in open(str(args.pred_file))]

    # Check that the lengths match
    if len(prediction) != len(key):
        print("Gold and prediction file must have same number of elements: %d in gold vs %d in prediction" % (len(key), len(prediction)))
        exit(1)
    
    # Score the predictions
    score(key, prediction, verbose=True)

