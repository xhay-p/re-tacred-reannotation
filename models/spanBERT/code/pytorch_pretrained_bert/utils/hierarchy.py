import json
import csv
import networkx as nx
from .constant import LABEL_TO_ID_HIER

ID_TO_LABEL = {val:key for key, val in LABEL_TO_ID_HIER.items()}

def get_hierarchical_relations():
    hier_rels = []
    with open('./../dataset/tacred/rel_list_hierarchically_aligned.csv', 'r') as fh:
        for rel in fh.readlines():
            hier_rels.append(rel.strip())
    hier_rels.sort()

    return hier_rels

def generate_graph():
    G = nx.Graph()
    hier_rels = get_hierarchical_relations()
    for line in hier_rels:
        if line == 'no_relation':
            continue
        else:
    #         print(line)
            relations = line.split('.')
            for i in range(len(relations)-1):
                # print((relations[i], LABEL_TO_ID_HIER[relations[i]], relations[i+1], LABEL_TO_ID_HIER[relations[i+1]]))
                G.add_edge(LABEL_TO_ID_HIER[relations[i]], LABEL_TO_ID_HIER[relations[i+1]])
    #         print("-----")

    return G

def get_lca(G, source, target):
    path1 = nx.shortest_path(G, source, LABEL_TO_ID_HIER['root'])
    path2 = nx.shortest_path(G, target, LABEL_TO_ID_HIER['root'])
    flag = 0
    for p in path1:
        for q in path2:
            if p == q:
                # print(p)
                flag = 1
                break
        if flag == 1:
            break
    # print(nx.shortest_path_length(G, source, p))
    return p

def get_parent_dictionary():
    parent_dict = {}
    G = generate_graph()

    for node in G.nodes:
        if nx.shortest_path_length(G, node, LABEL_TO_ID_HIER['root']) == 5:
            parent_dict[list(G.edges(node))[0][0]] = list(G.edges(node))[0][1]
    return parent_dict


def common_path_score(G, label, pred):
    lca = get_lca(G, label, pred)
    score = nx.shortest_path_length(G, lca, LABEL_TO_ID_HIER['root']) / nx.shortest_path_length(G, label, LABEL_TO_ID_HIER['root'])
    # print(score)
    return score, nx.shortest_path_length(G, label, pred)

def get_dis_lca_matrix():
    distance_matrix = []
    lca_matrix = []
    G = generate_graph()

    nodes = list(G.nodes)
    nodes.sort()

    for source in nodes:
        distance_vector = []
        lca_vector = []
        for target in nodes:
            distance_vector.append(nx.shortest_path_length(G, source, target))
            
            lca = get_lca(G, source, target)
            lca_vector.append(lca)
        distance_matrix.append(distance_vector)
        lca_matrix.append(lca_vector)

    return distance_matrix, lca_matrix