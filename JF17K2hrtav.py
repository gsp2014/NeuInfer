import tensorflow as tf
import numpy as np
import json
import time
ISOTIMEFORMAT="%Y-%m-%d %X"

tf.flags.DEFINE_string("data_dir", "./data", "The data dir.")
tf.flags.DEFINE_string("sub_dir", "JF17K_version1", "The sub data dir.")

FLAGS = tf.flags.FLAGS  
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))

def get_rel_ent():
    """
    Get a dict, each key is a relation, and the corresponding value is a list, where the i-th element is the set of the distinct attribute values of the relation's i-th attribute.
    """
    rel_entlist = {}
    with open(FLAGS.data_dir+"/"+FLAGS.sub_dir+"/train.txt", "r") as f:
        for line in f:
            line = line.strip().split("\t")
            rel = line[0]
            ents = line[1:]
            if rel not in rel_entlist:
                attrEntity = []
                for i in range(len(ents)):
                    attrEntity.append(set())
                rel_entlist[rel] = attrEntity
            for i in range(len(ents)):
                rel_entlist[rel][i].add(ents[i])
    return rel_entlist

def get_rel_ordered_entnum():
    """
    Get a dict, each key is a relation, and the corresponding value is an array of sorted attribute indexes in descending order, according to the number of the distinct values of the attributes.
    """
    rel_entlist = get_rel_ent()
    rel_ordered_entnum = {}
    for rel in rel_entlist:
        entnum = []
        for i in range(len(rel_entlist[rel])):
            entnum.append(len(rel_entlist[rel][i]))
        rel_ordered_entnum[rel] = np.argsort(-np.array(entnum))
    return rel_ordered_entnum

def write_json(rel_ordered_entnum, t_t):
    g = open(FLAGS.data_dir+"/"+FLAGS.sub_dir+"/n-ary_"+t_t+".json", "w")
    with open(FLAGS.data_dir+"/"+FLAGS.sub_dir+"/"+t_t+".txt", "r") as f:
        for line in f:
            line = line.strip().split("\t")
            if t_t == "train":
                begin = 1
            elif t_t == "test":
                begin = 2
            rel = line[begin-1]
            ents = line[begin:]
            if len(rel_ordered_entnum[rel]) != len(ents):
                print(line, "error rel_ordered_entnum:", rel_ordered_entnum[rel])
            xx_dict = {}
            xx_dict["H"] = ents[rel_ordered_entnum[rel][0]]
            xx_dict["R"] = rel+"0"
            xx_dict["T"] = ents[rel_ordered_entnum[rel][1]]
            for i in range(2, len(ents)):
                xx_dict[rel+str(i-1)] = ents[rel_ordered_entnum[rel][i]]
            xx_dict["N"] = len(ents)
            json.dump(xx_dict, g)
            g.write("\n")
    g.close()

if __name__ == "__main__":
    print(time.strftime(ISOTIMEFORMAT, time.localtime()))
    rel_ordered_entnum = get_rel_ordered_entnum()
    g = open(FLAGS.data_dir+"/"+FLAGS.sub_dir+"/n-ary_valid.json", "w")
    g.close()
    arr = ["train", "test"]
    for i in arr:
        write_json(rel_ordered_entnum, i)
    print(time.strftime(ISOTIMEFORMAT, time.localtime()))
