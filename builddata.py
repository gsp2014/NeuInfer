import tensorflow as tf
import numpy as np
import pickle
import time
ISOTIMEFORMAT="%Y-%m-%d %X"

tf.flags.DEFINE_string("data_dir", "./data", "The data dir.")
tf.flags.DEFINE_string("sub_dir", "WikiPeople", "The sub data dir.")
tf.flags.DEFINE_string("dataset_name", "WikiPeople", "The name of the dataset.")
tf.flags.DEFINE_string("bin_postfix", "", "The new postfix for the output bin file.")
tf.flags.DEFINE_boolean("if_permutate", False, "If permutate for test filter.")

FLAGS = tf.flags.FLAGS  
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))

def permutations(arr, position, end, res):
    """
    Permutate the array
    """
    if position == end:
        res.append(tuple(arr))                                        
    else:
        for index in range(position, end):
            arr[index], arr[position] = arr[position], arr[index]
            permutations(arr, position+1, end, res)
            arr[index], arr[position] = arr[position], arr[index]
    return res
    
def load_data_from_txt(filenames, entities_indexes = None, relations_indexes = None, ary_permutation = None):
    """
    Take a list of file names and build the corresponding dictionnary of facts
    """
    if entities_indexes is None:
        entities_indexes = dict()
        entities = set()
        next_ent = 0
    else:
        entities = set(entities_indexes)
        next_ent = max(entities_indexes.values()) + 1

    if relations_indexes is None:
        relations_indexes = dict()
        next_rel = 0
    else:
        next_rel = max(relations_indexes.values()) + 1
    if ary_permutation is None:
        ary_permutation= dict()

    max_n = 2  # The maximum arity of the facts
    for filename in filenames:
        with open(filename, "r") as f:
            for i, line in enumerate(f):
                xx_dict = eval(line)
                xx = xx_dict["N"]
                if xx > max_n:
                    max_n = xx
    data = []
    for i in range(max_n-1):
        data.append(dict())

    for filename in filenames:
        with open(filename) as f:
            lines = f.readlines()

        for _, line in enumerate(lines):
            aline = ()        
            xx_dict = eval(line)
            # parse the primary triple
            sub = xx_dict["H"]
            rel = xx_dict["R"]
            obj = xx_dict["T"]            
            if sub in entities:
                sub_ind = entities_indexes[sub]
            else:
                sub_ind = next_ent
                next_ent += 1
                entities_indexes[sub] = sub_ind
                entities.add(sub)                
            if obj in entities:
                obj_ind = entities_indexes[obj]
            else:
                obj_ind = next_ent
                next_ent += 1
                entities_indexes[obj] = obj_ind
                entities.add(obj)                
            if rel in relations_indexes:
                rel_ind = relations_indexes[rel]
            else:
                rel_ind = next_rel
                next_rel += 1
                relations_indexes[rel] = rel_ind
            aline = aline + (sub_ind,)
            aline = aline + (rel_ind,)
            aline = aline + (obj_ind,)
            # parse the auxiliary description(s)
            for k in xx_dict:
                if k == "H" or k == "T" or k == "R" or k == "N":
                    continue
                if k in relations_indexes:
                    attr_ind = relations_indexes[k]
                else:
                    attr_ind = next_rel
                    next_rel += 1
                    relations_indexes[k] = attr_ind
                if type(xx_dict[k]) == str:
                    val = xx_dict[k]
                    if val in entities:
                        val_ind = entities_indexes[val]
                    else:
                        val_ind = next_ent
                        next_ent += 1
                        entities_indexes[val] = val_ind
                        entities.add(val)
                    aline = aline + (attr_ind,)
                    aline = aline + (val_ind,)
                else:
                    for val in xx_dict[k]:  # Multiple attribute values
                        if val in entities:
                            val_ind = entities_indexes[val]
                        else:
                            val_ind = next_ent
                            next_ent += 1
                            entities_indexes[val] = val_ind
                            entities.add(val)
                        aline = aline + (attr_ind,)
                        aline = aline + (val_ind,)

            # Permutate the elements in the fact for negative sampling or further computing the filtered metrics in the test process
            flag = 0
            if FLAGS.if_permutate == True and xx_dict["N"] > 3:
                flag = 1
            if flag == 0:
                data[xx_dict["N"]-2][aline] = [1]
            else:
                if xx_dict["N"] in ary_permutation:
                    res = ary_permutation[xx_dict["N"]]
                else:
                    res = []
                    arr = np.array(range(xx_dict["N"]-2))
                    res = permutations(arr, 0, len(arr), res)
                    ary_permutation[xx_dict["N"]] = res
                for tpl in res:
                    tmpline = ()
                    tmpline = tmpline + (sub_ind,)
                    tmpline = tmpline + (rel_ind,)
                    tmpline = tmpline + (obj_ind,)
                    for tmp_ind in tpl:
                        tmpline = tmpline + (aline[3+2*tmp_ind], aline[3+2*tmp_ind+1])
                    data[xx_dict["N"]-2][tmpline] = [1]

    return data, entities_indexes, relations_indexes, ary_permutation

def get_neg_candidate_set(folder, entities_indexes, relations_indexes):
    """
    Get negative candidate set for replacing value
    """
    rel_head = {}
    rel_tail = {}
    attr_val = {} 
    with open(folder + "n-ary_train.json") as f:
        lines = f.readlines()
    for _, line in enumerate(lines):
        n_dict = eval(line)
        head = entities_indexes[n_dict["H"]]
        rel = relations_indexes[n_dict["R"]]
        tail = entities_indexes[n_dict["T"]]
        if rel not in rel_head:
            rel_head[rel] = []
        if head not in rel_head[rel]:
            rel_head[rel].append(head)
        if rel not in rel_tail:
            rel_tail[rel] = []
        if tail not in rel_tail[rel]:
            rel_tail[rel].append(tail)

        for k in n_dict:
            if k == "H" or k == "T" or k == "R" or k == "N":
                continue
            k_ind = relations_indexes[k]
            if k_ind not in attr_val:
                attr_val[k_ind] = []
            v = n_dict[k]
            if type(v) == str:
                v_ind = entities_indexes[v]
                if v_ind not in attr_val[k_ind]:
                    attr_val[k_ind].append(v_ind)
            else:  # Multiple attribute values
                for val in v:
                    val_ind = entities_indexes[val]
                    if val_ind not in attr_val[k_ind]:
                        attr_val[k_ind].append(val_ind)
    return rel_head, rel_tail, attr_val

def build_data(folder="data/", dataset_name="WikiPeople"):
    """
    Build data and save to files
    """
    train_facts, entities_indexes, relations_indexes, ary_permutation = load_data_from_txt([folder + "n-ary_train.json"])
    valid_facts, entities_indexes, relations_indexes, ary_permutation = load_data_from_txt([folder + "n-ary_valid.json"], entities_indexes = entities_indexes , relations_indexes = relations_indexes, ary_permutation = ary_permutation)
    test_facts, entities_indexes, relations_indexes, ary_permutation = load_data_from_txt([folder + "n-ary_test.json"], entities_indexes = entities_indexes , relations_indexes = relations_indexes, ary_permutation = ary_permutation)

    data_info = {}
    data_info["train_facts"] = train_facts
    data_info["valid_facts"] = valid_facts
    data_info["test_facts"] = test_facts
    data_info["entities_indexes"] = entities_indexes
    data_info["relations_indexes"] = relations_indexes
    if FLAGS.if_permutate == False:
        rel_head, rel_tail, attr_val = get_neg_candidate_set(folder, entities_indexes, relations_indexes)
        data_info["attr_val"] = attr_val
        data_info["rel_head"] = rel_head
        data_info["rel_tail"] = rel_tail
    with open(folder + dataset_name + FLAGS.bin_postfix + ".bin", "wb") as f:
        pickle.dump(data_info, f)

if __name__ == "__main__":
    print(time.strftime(ISOTIMEFORMAT, time.localtime()))
    afolder = FLAGS.data_dir + "/"
    if FLAGS.sub_dir != "":
        afolder = FLAGS.data_dir + "/" + FLAGS.sub_dir + "/"
    build_data(folder=afolder, dataset_name=FLAGS.dataset_name)
    print(time.strftime(ISOTIMEFORMAT, time.localtime()))
