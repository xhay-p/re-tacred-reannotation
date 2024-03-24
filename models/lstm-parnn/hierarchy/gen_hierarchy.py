import pickle
import random
from utils import constant

f = open("hierarchy/rel_list_hierarchically_aligned.csv", "r")
new_lines = f.read().splitlines()
lines = new_lines[1:]
random.seed(100)
random.shuffle(lines)
lines.insert(0, "root.no_relation")


label_to_id = {
    'per:family': -17,
    'per:location_of_residence': -16,
    'per:location_of_death': -15,
    'per:location_of_birth': -14,
    'org:location_of_headquarters': -13,
    'per-per': -12,
    'per-org': -11,
    'per-misc': -10,
    'per-loc': -9,
    'org-per': -8,
    'org-org': -7,
    'org-misc': -6,
    'org-loc': -5,
    'per': -4,
    'org': -3,
    'relation': -2,
    'root': -1,
    'no_relation': 0,
    'per:title': 1,
    'org:top_members/employees': 2,
    'per:employee_of': 3,
    'org:alternate_names': 4,
    'org:country_of_headquarters': 5,
    'per:countries_of_residence': 6,
    'org:city_of_headquarters': 7,
    'per:cities_of_residence': 8,
    'per:age': 9,
    'per:stateorprovinces_of_residence': 10,
    'per:origin': 11,
    'org:subsidiaries': 12,
    'org:parents': 13,
    'per:spouse': 14,
    'org:stateorprovince_of_headquarters': 15,
    'per:children': 16,
    'per:other_family': 17,
    'per:alternate_names': 18,
    'org:members': 19,
    'per:siblings': 20,
    'per:schools_attended': 21,
    'per:parents': 22,
    'per:date_of_death': 23,
    'org:member_of': 24,
    'org:founded_by': 25,
    'org:website': 26,
    'per:cause_of_death': 27,
    'org:political/religious_affiliation': 28,
    'org:founded': 29,
    'per:city_of_death': 30,
    'org:shareholders': 31,
    'org:number_of_employees/members': 32,
    'per:date_of_birth': 33,
    'per:city_of_birth': 34,
    'per:charges': 35,
    'per:stateorprovince_of_death': 36,
    'per:religion': 37,
    'per:stateorprovince_of_birth': 38,
    'per:country_of_birth': 39,
    'org:dissolved': 40,
    'per:country_of_death': 41
}



def gen_stnd_hierarchy():
    hierarchy = {}

    for line in lines:
        labels = line.split(".")
        for i in range(1, len(labels)):
            # print(labels[i-1], labels[i])
            hierarchy[label_to_id[labels[i]]] = label_to_id[labels[i - 1]]

    # print(hierarchy)

    file = open('hierarchy/hierarchy.pkl', 'wb')
    pickle.dump(hierarchy, file)
    file.close()
    gen_dist_matrix(label_to_id, hierarchy)

    return hierarchy


# file2 = open('hierarchy.pkl', 'rb')
# hierarchy = pickle.load(file2)
# file2.close()
# print(hierarchy)

# May need to modify to compress further
def gen_compressed_labels():
    label_id_map = {}
    label_id_map['no_relation'] = 0
    load_relation_map = {}
    compressed_hierarchy = {}
    cnt = 1
    for line in lines[1:]:
        labels = line.split(".")
        if label_to_id[labels[4]] < 0:
            load_relation_map[labels[5]] = labels[4]
        if labels[4] in label_id_map:
            continue
        label_id_map[labels[4]] = cnt
        cnt += 1

    for line in lines:
        labels = line.split(".")
        for i in range(1, min(5, len(labels))):
            if labels[i] in label_id_map:
                compressed_hierarchy[label_id_map[labels[i]]] = label_to_id[labels[i - 1]]
            else:
                compressed_hierarchy[label_to_id[labels[i]]] = label_to_id[labels[i - 1]]

    # print(load_relation_map)
    # print(compressed_hierarchy)
    # for k, v in label_id_map.items():
    # 	print(v,k)

    file = open('hierarchy/hierarchy.pkl', 'wb')
    pickle.dump(compressed_hierarchy, file)
    file.close()
    gen_dist_matrix(label_id_map, compressed_hierarchy)
    return label_id_map


def gen_dist_matrix(label_id_map, hierarchy):
    hierarchy_distances = {}
    lca = {}
    depth = {}

    for node in label_id_map.values():
        for target in label_id_map.values():
            if node < 0 or target < 0:
                continue
            distance = 1
            cur_input, cur_target = node, target
            while cur_input != cur_target:
                current_count = distance
                while cur_target != label_to_id["root"]:
                    cur_target = hierarchy[cur_target]
                    current_count = current_count + 1
                    if cur_input == cur_target:
                        break
                if cur_input == cur_target:
                    distance = current_count
                else:
                    cur_input = hierarchy[cur_input]
                    distance = distance + 1
                    cur_target = target
            hierarchy_distances[node, target] = distance
            lca[node, target] = cur_input

    for rel,node in label_to_id.items():
        if rel in label_id_map or node < 0:

            if rel in label_id_map:
                node = label_id_map[rel]

            cur_node = node
            distance = 1
            while cur_node != label_to_id["root"]:
                distance += 1
                cur_node = hierarchy[cur_node]
            depth[node] = distance

    # print(depth)
    # print(lca)

    file = open('hierarchy/hierarchy_distances.pkl', 'wb')
    pickle.dump(hierarchy_distances, file)
    pickle.dump(lca, file)
    pickle.dump(depth, file)
    file.close()
    # print(hierarchy_distances)


# file2 = open('hierarchy.pkl', 'rb')
# hierarchy = pickle.load(file2)
# file2.close()
# print(hierarchy)

# gen_stnd_hierarchy()
# gen_compressed_labels()
