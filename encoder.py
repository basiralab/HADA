from __future__ import division
from __future__ import print_function
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix

import numpy as np
import os
# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

import tensorflow as tf
import settings
from constructor import get_placeholder, get_model, format_data_new, get_optimizer, update
# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS

class Encoder():
    def __init__(self, settings):
        self.iteration = settings['iterations']
        self.model = settings['model']
        

    def erun(self, adj, features):
        tf.reset_default_graph()
        model_str = self.model
        # formatted data
        feas = format_data_new(adj, coo_matrix(features))
        
        # Define placeholders
        placeholders = get_placeholder(feas['adj'])

        # construct model
        d_real, discriminator, ae_model = get_model(model_str, placeholders, feas['num_features'], feas['num_nodes'], feas['features_nonzero'])

        # Optimizer
        opt = get_optimizer(model_str, ae_model, discriminator, placeholders, feas['pos_weight'], feas['norm'], d_real, feas['num_nodes'])

        # Initialize session
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # Train model
        for epoch in range(self.iteration):

            emb, avg_cost = update(ae_model, opt, sess, feas['adj_norm'], feas['adj_label'], feas['features'], placeholders, feas['adj'], features)
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost))
            if (epoch+1) == 5:
                break
                
        return emb
