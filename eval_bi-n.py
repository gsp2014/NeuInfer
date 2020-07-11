import tensorflow as tf
import numpy as np
np.random.seed(1234)
import os
import pickle
from importlib import import_module
from multiprocessing import JoinableQueue, Queue, Process
from log import Logger
from batching import *

tf.flags.DEFINE_string("data_dir", "./data", "The data dir.")
tf.flags.DEFINE_string("sub_dir", "WikiPeople", "The sub data dir.")
tf.flags.DEFINE_string("dataset_name", "WikiPeople", "The name of the dataset.")
tf.flags.DEFINE_string("wholeset_name", "WikiPeople_permutate", "The name of the whole dataset for negative sampling or computing the filtered metrics.")
tf.flags.DEFINE_string("model_name", "WikiPeople", "")
tf.flags.DEFINE_integer("embedding_dim", 100, "The embedding dimension.")
tf.flags.DEFINE_integer("hrtFCNs_layers", 1, "The number of layers in hrt-FCNs")
tf.flags.DEFINE_integer("hrtavFCNs_layers", 1, "The number of layers in hrtav-FCNs")
tf.flags.DEFINE_integer("g_theta_dim", 1000, "The dimension of the interaction vector o_hrtav.")
tf.flags.DEFINE_float("weight", 0.3, "The weight factor of the scores")
tf.flags.DEFINE_integer("batch_size", 128, "The batch size.")
tf.flags.DEFINE_boolean("is_trainable", False, "")
tf.flags.DEFINE_integer("n_epochs", 5000, "The number of evaluation epochs.")
tf.flags.DEFINE_integer("start_epoch", 100, "Evaluate the model from start_epoch.")
tf.flags.DEFINE_integer("evalStep", 100, "Evaluate the model every evalStep.")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement.")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices.")
tf.flags.DEFINE_integer("metric_num", 4, "")
tf.flags.DEFINE_integer("valid_or_test", 1, "valid: 1, test: 2")
tf.flags.DEFINE_string("model_postfix", "", "Which model to load.")
tf.flags.DEFINE_string("gpu_ids", "0,1,2,3", "Comma-separated gpu ids.")
tf.flags.DEFINE_string("run_folder", "./", "The dir to store models.")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
model = import_module("model"+FLAGS.model_postfix)
gpu_ids = list(map(int, FLAGS.gpu_ids.split(",")))

# The log file to store the parameters and the evaluation details of each epoch
logger = Logger("logs", str(FLAGS.valid_or_test)+"_evalres_"+FLAGS.model_name+"_"+str(FLAGS.embedding_dim)+"_"+str(FLAGS.hrtFCNs_layers)+"_"+str(FLAGS.hrtavFCNs_layers)+"_"+str(FLAGS.g_theta_dim)+"_"+str(FLAGS.weight)+"_"+str(FLAGS.batch_size)).logger
logger.info("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    logger.info("{}={}".format(attr.upper(), value))

# Load validation and test data
logger.info("Loading data...")
afolder = FLAGS.data_dir + "/"
if FLAGS.sub_dir != "":
    afolder = FLAGS.data_dir + "/" + FLAGS.sub_dir + "/"
with open(afolder + FLAGS.dataset_name + ".bin", "rb") as fin:
    data_info = pickle.load(fin)
valid = data_info["valid_facts"]
test = data_info["test_facts"]
entities_indexes = data_info["entities_indexes"]
relations_indexes = data_info["relations_indexes"]
entity_array = np.array(list(entities_indexes.values()))
relation_array = np.array(list(relations_indexes.values()))

# Load the whole dataset for computing the filtered metrics
with open(afolder + FLAGS.wholeset_name + ".bin", "rb") as fin:
    data_info1 = pickle.load(fin)
whole_train = data_info1["train_facts"]
whole_valid = data_info1["valid_facts"]
whole_test = data_info1["test_facts"]
logger.info("Loading data... finished!")
logger.info("Size of entity set and relation set: "+str(len(entities_indexes))+", "+str(len(relations_indexes)))

# Prepare validation and test facts
x_valid = []
y_valid = []
for k in valid:
    x_valid.append(np.array(list(k.keys())).astype(np.int32))
    y_valid.append(np.array(list(k.values())).astype(np.float32))
x_test = []         
y_test = []         
for k in test:      
    x_test.append(np.array(list(k.keys())).astype(np.int32))
    y_test.append(np.array(list(k.values())).astype(np.int32))

# Output directory for models and checkpoint directory
out_dir = os.path.abspath(os.path.join(FLAGS.run_folder, "runs", FLAGS.model_name))
checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
checkpoint_prefix = os.path.join(checkpoint_dir, "model")
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

class Predictor(Process):
    """
    Predictor for evaluation
    """
    def __init__(self, in_queue, out_queue, epoch, gpu_id):
        Process.__init__(self)
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.epoch = epoch
        self.gpu_id = gpu_id
    def run(self):
        # set GPU id before importing tensorflow!
        os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(self.gpu_id)
        # import tensorflow here
        import tensorflow as tf
        session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement, log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = True
        sess = tf.Session(config=session_conf)
        aNeuInfer = model.NeuInfer(
            n_entities=len(entities_indexes), 
            n_relations=len(relations_indexes), 
            embedding_dim=FLAGS.embedding_dim, 
            hrtFCNs_layers=FLAGS.hrtFCNs_layers, 
            hrtavFCNs_layers=FLAGS.hrtavFCNs_layers, 
            g_theta_dim=FLAGS.g_theta_dim, 
            weight=FLAGS.weight, 
            batch_size=FLAGS.batch_size,
            is_trainable=FLAGS.is_trainable)

        _file = checkpoint_prefix + "-" + str(self.epoch)
        aNeuInfer.saver.restore(sess, _file)

        while True:
            dat = self.in_queue.get()
            if dat is None:
                self.in_queue.task_done()
                break
            else:
                (x_batch, y_batch, arity, ind) = dat
                feed_dict = {
                    aNeuInfer.input_x: x_batch,
                    aNeuInfer.input_y: y_batch,
                    aNeuInfer.arity: arity
                }
                scores, loss = sess.run([aNeuInfer.final_score, aNeuInfer.loss], feed_dict)
                self.out_queue.put((scores, loss, ind))
                self.in_queue.task_done()
        sess.close()
        return

def eval_one(x_batch, y_batch, evaluation_queue, result_queue, data_index, pred_ind=0):
    """
    Predict the pred_ind-th element (h/r/t/attribute/attribute value) of each fact in x_batch
    """
    mrr = 0.0
    hits1 = 0.0
    hits3 = 0.0
    hits10 = 0.0
    total_loss = 0.0
    for i in range(len(x_batch)):
        if pred_ind % 2 == 1:  # predict relation or attribute
            tmp_array = relation_array
            right_index = np.argwhere(relation_array == x_batch[i][pred_ind])[0][0]
        else:
            tmp_array = entity_array  # predict entity or attribute value
            right_index = np.argwhere(entity_array == x_batch[i][pred_ind])[0][0]
        new_x_batch = np.tile(x_batch[i], (len(tmp_array), 1))
        new_x_batch[:, pred_ind] = tmp_array
        new_y_batch = np.tile(np.array([0]).astype(np.int32), (len(tmp_array), 1))
        new_y_batch[right_index] = [1]
        while len(new_x_batch) % FLAGS.batch_size != 0:
            new_x_batch = np.append(new_x_batch, [x_batch[i]], axis=0)
            new_y_batch = np.append(new_y_batch, [y_batch[i]], axis=0)
        tmp_array1 = new_x_batch[:, pred_ind]
        listIndexes = range(0, len(new_x_batch), FLAGS.batch_size)
        nn = len(listIndexes)
        results = []
        tmp_res_list = []
        for tmpIndex in range(nn):
            tmp_res_list.append([])
        arity = int(len(x_batch[i])/2) + 1  #ary2: len=3, ary3: len=5
        for tmpIndex in range(nn - 1):
            evaluation_queue.put((new_x_batch[listIndexes[tmpIndex]:listIndexes[tmpIndex + 1]],
                new_y_batch[listIndexes[tmpIndex]:listIndexes[tmpIndex + 1]], arity, tmpIndex))
        evaluation_queue.put((new_x_batch[listIndexes[-1]:], new_y_batch[listIndexes[-1]:], arity, nn-1))
        evaluation_queue.join()

        for tmp_id in range(nn):
            (res, loss, ind) = result_queue.get()
            tmp_res_list[ind] = res
            total_loss = total_loss + loss
        for tmp_id in range(nn):
            results = np.append(results, tmp_res_list[tmp_id])

        results = np.reshape(results, [tmp_array1.shape[0], 1])
        results_with_id = np.hstack(
            (np.reshape(tmp_array1, [tmp_array1.shape[0], 1]), results))
        results_with_id = results_with_id[np.argsort(-results_with_id[:, 1])]
        results_with_id = results_with_id[:, 0].astype(int)
        _filter = 0
        for tmpxx in results_with_id:
            if tmpxx == x_batch[i][pred_ind]:
                break
            tmp_list = list(x_batch[i])
            tmp_list[pred_ind] = tmpxx
            tmpTriple = tuple(tmp_list)
            if (len(whole_train) > data_index) and (tmpTriple in whole_train[data_index]):
                continue
            elif (len(whole_valid) > data_index) and (tmpTriple in whole_valid[data_index]):
                continue
            elif (len(whole_test) > data_index) and (tmpTriple in whole_test[data_index]):
                continue
            else:
                _filter += 1

        mrr += 1.0 / (_filter + 1)
        if _filter < 10:
            hits10 += 1
            if _filter < 3:
                hits3 += 1
                if _filter < 1:
                    hits1 += 1
    return np.array([total_loss, mrr, hits10, hits3, hits1])

def eval_all(epoch, x_test, y_test, evaluation_queue, result_queue):
    """
    Predict all the elements of each fact in the whole set x_test
    """
    rel_results = np.zeros(FLAGS.metric_num)
    ent_results = np.zeros(FLAGS.metric_num)
    rel_c = 0
    ent_c = 0
    all_loss = 0.0
    len_data = 0
    rel_results_list = []
    ent_results_list = []
    rel_c_list = []
    ent_c_list = []
    for i in range(len(x_test)):
        rel_results_list.append(np.zeros(FLAGS.metric_num))
        ent_results_list.append(np.zeros(FLAGS.metric_num))
        rel_c_list.append(0)
        ent_c_list.append(0)
    for i in range(len(x_test)):
        if len(x_test[i]) == 0:
            continue
        len_data = len_data + len(x_test[i])
        #n_ary = i+2  #2-ary in index 0
        n_ary = int(len(x_test[i][0])/2) + 1  # ary2: len=3, ary3: len=5
        if epoch == FLAGS.n_epochs: 
            for j in range(2*n_ary-1):
                tmp = eval_one(x_test[i], y_test[i], evaluation_queue, result_queue, i, j)
                tmp_results = tmp[1:]
                all_loss = all_loss + tmp[0]
                if j % 2 == 1:  # h(0), r, t(2), a1, v1(4), a2, v2(6), ...
                    rel_results = rel_results + tmp_results
                    rel_c = rel_c + len(x_test[i])
                    rel_results_list[i] = rel_results_list[i] + tmp_results
                    rel_c_list[i] = rel_c_list[i] + len(x_test[i])
                else:
                    ent_results = ent_results + tmp_results
                    ent_c = ent_c + len(x_test[i])
                    ent_results_list[i] = ent_results_list[i] + tmp_results
                    ent_c_list[i] = ent_c_list[i] + len(x_test[i])
        else:  # If it is not the last epoch, only predict values
            for j in range(2*n_ary-1):
                if j % 2 == 1:
                    continue
                tmp = eval_one(x_test[i], y_test[i], evaluation_queue, result_queue, i, j)
                tmp_results = tmp[1:]
                all_loss = all_loss + tmp[0]
                ent_results = ent_results + tmp_results
                ent_c = ent_c + len(x_test[i])
                ent_results_list[i] = ent_results_list[i] + tmp_results
                ent_c_list[i] = ent_c_list[i] + len(x_test[i])

    for i in range(len(gpu_ids)):
        evaluation_queue.put(None)
    logger.info(FLAGS.dataset_name+", len(data): "+str(len_data))
    logger.info("epoch: "+str(epoch)+", testloss: "+str(all_loss/(rel_c+ent_c)))
    logger.info("result lists: ent_results_list ent_c_list rel_results_list rel_c_list")
    logger.info(str(ent_results_list))
    logger.info(str(ent_c_list))
    logger.info(str(rel_results_list))
    logger.info(str(rel_c_list))
    if rel_c == 0:
        rel_c = 1
    logger.info("epoch: "+str(epoch)+", relation_entity: "+str(rel_results/rel_c)+"; "+str(ent_results/ent_c))
    logger.info("predict entities:")
    for i in range(len(ent_c_list)):
        logger.info("arity "+str(i+2)+": "+str(np.array(ent_results_list[i])/ent_c_list[i]))
    logger.info("epoch: "+str(epoch)+", res_on_total"+str(np.sum(ent_results_list, axis=0)/np.sum(ent_c_list, axis=0)))
    logger.info("epoch: "+str(epoch)+", ent_results_bi ent_results_n"+str(ent_results_list[0]/ent_c_list[0])+str( (np.sum(ent_results_list, axis=0)-ent_results_list[0]) / (np.sum(ent_c_list, axis=0)-ent_c_list[0]) ))
    if epoch == FLAGS.n_epochs:
        logger.info("predict relations:")
        for i in range(len(ent_c_list)):
            logger.info("arity "+str(i+2)+": "+str(np.array(rel_results_list[i])/rel_c_list[i]))
        logger.info("epoch: "+str(epoch)+", res_on_total"+str(np.sum(rel_results_list, axis=0)/np.sum(rel_c_list, axis=0)))
        logger.info("epoch: "+str(epoch)+", rel_results_bi re_results_n"+str(rel_results_list[0]/rel_c_list[0])+str( (np.sum(rel_results_list, axis=0)-rel_results_list[0]) / (np.sum(rel_c_list, axis=0)-rel_c_list[0]) ))

def check_epoch_finish(model_dir, epoch):
    """
    Check if the epoch training finishes 
    """
    for root, dirs, files in os.walk(model_dir):
        for name in files:
            if name.find(str(epoch)+".") != -1:
                return True
    return False

if __name__ == "__main__":
    cur_epoch = FLAGS.start_epoch
    while True:
        if check_epoch_finish(checkpoint_dir, cur_epoch) == True:
            logger.info("begin eval"+str(cur_epoch))
            evaluation_queue = JoinableQueue()
            result_queue = Queue()
            p_list = []
            for i in range(len(gpu_ids)):
                p = Predictor(evaluation_queue, result_queue, cur_epoch, gpu_ids[i])
                p_list.append(p)
            for p in p_list:
                p.start()
            if FLAGS.valid_or_test == 1:
                eval_all(cur_epoch, x_valid, y_valid, evaluation_queue, result_queue)
            else:
                eval_all(cur_epoch, x_test, y_test, evaluation_queue, result_queue)
            for p in p_list:
                p.join()
            logger.info("finish eval"+str(cur_epoch))
            cur_epoch = cur_epoch + FLAGS.evalStep
            if cur_epoch > FLAGS.n_epochs:
                break
    exit()
