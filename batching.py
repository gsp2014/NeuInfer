import numpy as np
import tensorflow as tf

def replace_ent(n_entities, last_idx, attr_val, rel_head, rel_tail, arity, new_facts_indexes, new_facts_values, whole_train_facts):
    """
    Replace entities or attribute values randomly to get negative samples
    """
    for cur_idx in range(last_idx):
        rel_ind = (np.random.randint(np.iinfo(np.int32).max) % arity) * 2 - 1  # 1, 3, 5, ...
        if rel_ind == -1:
            rel_ind = 1
            rmd_dict = rel_head
            shift = -1
        elif rel_ind == 1:
            rmd_dict = rel_tail
            shift = 1
        else:
            rmd_dict = attr_val
            shift = 1
        tmp_rel = new_facts_indexes[last_idx + cur_idx, rel_ind]
        tmp_len = len(rmd_dict[tmp_rel])
        rdm_w = np.random.randint(0, tmp_len)  # [low,high)

        # Sample a random entity or attribute value
        times = 1
        tmp_array = new_facts_indexes[last_idx + cur_idx]
        tmp_array[rel_ind + shift] = rmd_dict[tmp_rel][rdm_w]
        while (tuple(tmp_array) in whole_train_facts):
            if (tmp_len == 1) or (times > 2*tmp_len) or (times > 100):
                tmp_array[rel_ind + shift] = np.random.randint(0, n_entities)
            else:
                rdm_w = np.random.randint(0, tmp_len)
                tmp_array[rel_ind + shift] = rmd_dict[tmp_rel][rdm_w]
            times = times + 1
        new_facts_indexes[last_idx + cur_idx, rel_ind + shift] = tmp_array[rel_ind + shift]
        new_facts_values[last_idx + cur_idx] = [0]

def replace_rel(n_relations, last_idx, arity, new_facts_indexes, new_facts_values, whole_train_facts):
    """
    Replace relations or attributes randomly to get negative samples
    """
    rdm_ws = np.random.randint(0, n_relations, last_idx)        
    for cur_idx in range(last_idx):
        rel_ind = (np.random.randint(np.iinfo(np.int32).max) % arity) * 2 - 1
        if rel_ind == -1:
            rel_ind = 1
        # Sample relation or an attribute randomly
        tmp_array = new_facts_indexes[last_idx + cur_idx]
        tmp_array[rel_ind] = rdm_ws[cur_idx]
        while (tuple(tmp_array) in whole_train_facts):
            tmp_array[rel_ind] = np.random.randint(0, n_relations)
        new_facts_indexes[last_idx + cur_idx, rel_ind] = tmp_array[rel_ind]
        new_facts_values[last_idx + cur_idx] = [0]

def Batch_Loader(train_batch_indexes, train_batch_values, entities_indexes, relations_indexes, attr_val, rel_head, rel_tail, batch_size, arity, whole_train_facts):
    new_facts_indexes = np.empty((batch_size*2, 2*arity - 1)).astype(np.int32)
    new_facts_values = np.empty((batch_size*2, 1)).astype(np.float32)

    idxs = np.random.randint(0, len(train_batch_values), batch_size)
    new_facts_indexes[:batch_size, :] = train_batch_indexes[idxs, :]
    new_facts_values[:batch_size] = train_batch_values[idxs, :]
    last_idx = batch_size
    
    # Copy everyting in advance
    new_facts_indexes[last_idx:(last_idx*2), :] = new_facts_indexes[:last_idx, :]
    new_facts_values[last_idx:(last_idx*2)] = new_facts_values[:last_idx]
    n_entities = len(entities_indexes)
    n_relations = len(relations_indexes)
    ent_rel = np.random.randint(np.iinfo(np.int32).max) % (n_entities+n_relations)
    if ent_rel < n_entities:  # 0~(n_entities-1)
        replace_ent(n_entities, last_idx, attr_val, rel_head, rel_tail, arity, new_facts_indexes, new_facts_values, whole_train_facts)
    else:
        replace_rel(n_relations, last_idx, arity, new_facts_indexes, new_facts_values, whole_train_facts)
    last_idx += batch_size

    return new_facts_indexes[:last_idx, :], new_facts_values[:last_idx]
