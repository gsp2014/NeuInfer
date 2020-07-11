import tensorflow as tf
import numpy as np
np.random.seed(1234)
import os
import pickle
from importlib import import_module
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
tf.flags.DEFINE_boolean("is_trainable", True, "")
tf.flags.DEFINE_float("learning_rate", 0.0001, "The learning rate.")
tf.flags.DEFINE_integer("n_epochs", 5000, "The number of training epochs.")
tf.flags.DEFINE_boolean("if_restart", False, "")
tf.flags.DEFINE_integer("start_epoch", 0, "Change this when restarting from halfway.")
tf.flags.DEFINE_integer("saveStep", 100, "Save the model every saveStep.")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement.")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices.")
tf.flags.DEFINE_string("model_postfix", "", "Which model to load.")
tf.flags.DEFINE_string("run_folder", "./", "The dir to store models.")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
model = import_module("model"+FLAGS.model_postfix)

# The log file to store the parameters and the training details of each epoch
logger = Logger("logs", "run_"+FLAGS.model_name+"_"+str(FLAGS.embedding_dim)+"_"+str(FLAGS.hrtFCNs_layers)+"_"+str(FLAGS.hrtavFCNs_layers)+"_"+str(FLAGS.g_theta_dim)+"_"+str(FLAGS.weight)+"_"+str(FLAGS.batch_size)+"_"+str(FLAGS.learning_rate)).logger
logger.info("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    logger.info("{}={}".format(attr.upper(), value))

# Load training data
logger.info("Loading data...")
afolder = FLAGS.data_dir + "/"
if FLAGS.sub_dir != "":
    afolder = FLAGS.data_dir + "/" + FLAGS.sub_dir + "/"
with open(afolder + FLAGS.dataset_name + ".bin", "rb") as fin:
    data_info = pickle.load(fin)
train = data_info["train_facts"]
entities_indexes = data_info["entities_indexes"]
relations_indexes = data_info["relations_indexes"]
attr_val = data_info["attr_val"]
rel_head = data_info["rel_head"]
rel_tail = data_info["rel_tail"]
entity_array = np.array(list(entities_indexes.values()))
relation_array = np.array(list(relations_indexes.values()))

# Load the whole dataset for negative sampling in "batching.py"
with open(afolder + FLAGS.wholeset_name + ".bin", "rb") as fin:
    data_info1 = pickle.load(fin)
whole_train = data_info1["train_facts"]
logger.info("Loading data... finished!")

with tf.Graph().as_default():
    tf.set_random_seed(1234)
    session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement, log_device_placement=FLAGS.log_device_placement)
    session_conf.gpu_options.allow_growth = True
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        aNeuInfer = model.NeuInfer(
            n_entities=len(entities_indexes), 
            n_relations=len(relations_indexes), 
            embedding_dim=FLAGS.embedding_dim, 
            hrtFCNs_layers=FLAGS.hrtFCNs_layers, 
            hrtavFCNs_layers=FLAGS.hrtavFCNs_layers, 
            g_theta_dim=FLAGS.g_theta_dim, 
            weight=FLAGS.weight, 
            batch_size=FLAGS.batch_size*2,
            is_trainable=FLAGS.is_trainable)
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        grads_and_vars = optimizer.compute_gradients(aNeuInfer.loss)
        train_op = optimizer.apply_gradients(grads_and_vars)
           
        # Output directory for models and summaries
        out_dir = os.path.abspath(os.path.join(FLAGS.run_folder, "runs", FLAGS.model_name))
        logger.info("Writing to {}\n".format(out_dir))
   
        # Train Summaries
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
           
        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
   
        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch, arity):
            """
            A single training step
            """
            feed_dict = {
                aNeuInfer.input_x: x_batch,
                aNeuInfer.input_y: y_batch,
                aNeuInfer.arity: arity,
            }
            _, loss = sess.run([train_op, aNeuInfer.loss], feed_dict)
            return loss
        
        # If restart for a certain epoch, then load the model
        if FLAGS.if_restart == True:
            _file = checkpoint_prefix + "-" + str(FLAGS.start_epoch)
            aNeuInfer.saver.restore(sess, _file)

        # Training
        n_batches_per_epoch = []
        for i in train:
            ll = len(i)
            if ll == 0:
                n_batches_per_epoch.append(0)
            else:
                n_batches_per_epoch.append(int((ll - 1) / FLAGS.batch_size) + 1)
        for epoch in range(FLAGS.start_epoch, FLAGS.n_epochs):
            train_loss = 0
            for i in range(len(train)):
                train_batch_indexes = np.array(list(train[i].keys())).astype(np.int32)
                train_batch_values = np.array(list(train[i].values())).astype(np.float32)
                for batch_num in range(n_batches_per_epoch[i]):
                    arity = i+2  # 2-ary in index 0
                    x_batch, y_batch = Batch_Loader(train_batch_indexes, train_batch_values, entities_indexes, relations_indexes, attr_val, rel_head, rel_tail, FLAGS.batch_size, arity, whole_train[i])
                    tmp_loss = train_step(x_batch, y_batch, arity)
                    train_loss = train_loss + tmp_loss
                
            logger.info("nepoch: "+str(epoch+1)+", trainloss: "+str(train_loss))
            if (epoch+1) % FLAGS.saveStep == 0:
                path = aNeuInfer.saver.save(sess, checkpoint_prefix, global_step=epoch+1)
                logger.info("Saved model checkpoint to {}\n".format(path))
        train_summary_writer.close
