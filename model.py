import tensorflow as tf
import numpy as np
import math
class NeuInfer(object):
    """
    The proposed NeuInfer model
    """
    def g_theta(self, o_i, o_j, scope='g_theta', reuse=tf.AUTO_REUSE):
        """
        g_theta: Obtain the interaction vector of the o_i (hrt) and o_j (aivi)
        """
        with tf.variable_scope(scope, reuse=reuse) as scope:
            if not reuse: print("g_theta warn:", scope.name)
            if self.hrtavFCNs_layers == 1:
                g_1 = tf.contrib.layers.fully_connected(tf.concat([o_i, o_j], axis=1), self.g_theta_dim, activation_fn=tf.nn.relu)
            else:  # hrtavFCNs_layers = 2
                dim1 = int((5*self.embedding_dim-self.g_theta_dim)/2)
                g_1 = tf.contrib.layers.fully_connected(tf.concat([o_i, o_j], axis=1), 5*self.embedding_dim-dim1, activation_fn=tf.nn.relu)
                g_1 = tf.contrib.layers.fully_connected(g_1, self.g_theta_dim, activation_fn=tf.nn.relu)
            return g_1
            
    def hrtavFCNs(self, i, o_hrtaivi_list):
        """
        hrtavFCNs: Obtain the interaction vectors of hrt and all the atribute-value pairs via g-FCN
        """
        a_embed = tf.reshape(tf.nn.embedding_lookup(self.rel_embeddings, self.input_x[:, i]), [-1, self.embedding_dim])
        v_embed = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings, self.input_x[:, i+1]), [-1, self.embedding_dim])
        av_embed = tf.concat([a_embed, v_embed], -1)
        hrt_av_compability = self.g_theta(self.hrtFCNs_res, av_embed)
        hrt_av_compability = tf.reshape(hrt_av_compability, [1, hrt_av_compability.get_shape().as_list()[0], hrt_av_compability.get_shape().as_list()[1]])
        o_hrtaivi_list = tf.cond(tf.equal(i, 3), 
            lambda:hrt_av_compability, 
            lambda:tf.concat([o_hrtaivi_list, hrt_av_compability], 0))
        i = i + 2 
        return i, o_hrtaivi_list
    
    def __init__(self, n_entities, n_relations, embedding_dim, hrtFCNs_layers=1, hrtavFCNs_layers=1, g_theta_dim=1000, weight=0.5, batch_size=128, is_trainable=True):
        # input_x: The input facts; input_y: The label of the input fact; arity: The arity of the input facts
        self.input_x = tf.placeholder(tf.int32, [batch_size, None], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [batch_size, 1], name="input_y")
        self.arity = tf.placeholder(tf.int32, name="arity")
        
        self.embedding_dim = embedding_dim
        self.hrtavFCNs_layers = hrtavFCNs_layers
        self.g_theta_dim = g_theta_dim
        self.batch_size = batch_size

        # -------- Embedding layer --------
        with tf.name_scope("embeddings"):
            bound = math.sqrt(1.0/embedding_dim)
            self.ent_embeddings = tf.Variable(tf.random_uniform([n_entities, embedding_dim], minval=-bound, maxval=bound), name="ent_embeddings")
            self.rel_embeddings = tf.Variable(tf.random_uniform([n_relations, embedding_dim], minval=-bound, maxval=bound), name="rel_embeddings")
        
        # -------- Validity evaluation component --------
        with tf.name_scope("validity"):
            self.h_embed = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings, self.input_x[:, 0]), [-1, embedding_dim])
            self.r_embed = tf.reshape(tf.nn.embedding_lookup(self.rel_embeddings, self.input_x[:, 1]), [-1, embedding_dim])
            concat_embed = tf.concat([self.h_embed, self.r_embed], -1)
            self.t_embed = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings, self.input_x[:, 2]), [-1, embedding_dim])
            self.hrt_embed = tf.concat([concat_embed, self.t_embed], -1)
            self.hrtFCNs_res = self.hrt_embed

            # hrt-FCNs
            if hrtFCNs_layers == 1:
                dim1 = int(3*self.embedding_dim/2)
                self.hrtFCNs_res1 = tf.contrib.layers.fully_connected(self.hrt_embed, 3*self.embedding_dim-dim1, activation_fn=tf.nn.relu)
            else:  # hrtFCNs_layers = 2
                self.hrtFCNs_res1 = tf.contrib.layers.fully_connected(self.hrt_embed, 2*self.embedding_dim, activation_fn=tf.nn.relu)
                self.hrtFCNs_res1 = tf.contrib.layers.fully_connected(self.hrtFCNs_res1, self.embedding_dim, activation_fn=tf.nn.relu)
            
            # FCN1
            self.hrt_scores = tf.contrib.layers.fully_connected(self.hrtFCNs_res1, self.input_y.get_shape()[1].value, activation_fn=None)
            self.validity = tf.nn.sigmoid(self.hrt_scores)

        # -------- Compatibility evaluation component --------
        with tf.name_scope("compatibility"):
            # get the results of hrtav-FCNs
            i = tf.constant(3, dtype=tf.int32)
            n = 2*self.arity - 1 
            o_hrtaivi_list = tf.zeros([1, batch_size, self.g_theta_dim], dtype=tf.float32)
            _, o_hrtaivi_list = tf.while_loop(cond=lambda i, o_hrtaivi_list:tf.less(i, n), 
                body=self.hrtavFCNs, loop_vars=[i, o_hrtaivi_list], 
                shape_invariants=[i.get_shape(), tf.TensorShape([None, o_hrtaivi_list.shape[1], o_hrtaivi_list.shape[2]])])
            self.o_hrtaivi_list = o_hrtaivi_list
            self.o_hrtav = tf.reduce_min(self.o_hrtaivi_list, axis=0)
            
            # FCN2
            self.compatibility = tf.contrib.layers.fully_connected(self.o_hrtav, self.input_y.get_shape()[1].value, activation_fn=None)
            self.compatibility = tf.nn.sigmoid(self.compatibility)

        # -------- Final score and loss function --------
        with tf.name_scope("output_loss"):
            self.final_score = tf.cond(tf.equal(self.arity, 2), 
                lambda:self.validity, 
                lambda:weight * self.validity + (1-weight) * self.compatibility)
            self.loss = -tf.reduce_mean(self.input_y * tf.log(tf.clip_by_value(self.final_score, 1e-10, 1.0)) + (1 - self.input_y) * tf.log(tf.clip_by_value(1 - self.final_score, 1e-10, 1.0)))
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=500)
