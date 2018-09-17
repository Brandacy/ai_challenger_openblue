"""
-------------------------------------------------
   File Name：     train
   Description :
   Author :       weiwanshun
   date：          2018/7/5
-------------------------------------------------
   Change Activity:
                   2018/7/5:
-------------------------------------------------
"""
__author__ = 'weiwanshun'

import time
import json
import numpy as np
import pandas as pd
import keras

# from models_generator import *
from models_generator import *

from check_point import ModelCheckpoint
from data_preprocess import get_train_data
import tensorflow as tf
from keras import backend as K
from sklearn.model_selection import train_test_split

num_cores = 4

config = tf.ConfigProto(intra_op_parallelism_threads=num_cores, inter_op_parallelism_threads=num_cores,
                        allow_soft_placement=True, device_count={'CPU': 4})
session = tf.Session(config=config)
K.set_session(session)

import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

config = tf.ConfigProto()
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
sess = tf.Session(config=config)

KTF.set_session(sess)


filters = '!?"#$%&()*+,-./:;<=>@[\\]^_`{|}~\t\n\r'
dic_path = "./output/"
model_path = "./model_files/"
if not os.path.exists(dic_path):
    os.makedirs(dic_path)


def train():
    input_length = 600
    x_padded_seqs, y_one_hot, word_size, y_num = get_train_data(input_length)

    x_train_padded_seqs, x_test_padded_seqs, y_train_one_hot, y_test_one_hot = train_test_split(x_padded_seqs,
                                                                                                y_one_hot,
                                                                                                test_size=0.15,
                                                                                                random_state=666)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    check_point = ModelCheckpoint(filepath='epoch{epoch:02d}_iou{iou:.2f}_valloss{val_loss:.2f}.hdf5',
                                  monitor='iou', verbose=1, weight_window=0.5, save_best_only=True,
                                  save_weights_only=False,
                                  mode='max', period=1)

    # check_point = ModelCheckpoint(filepath=model_path + time.strftime("%Y%m%d%H%M%S",
    #                                                                   time.localtime()) + '##epoch{epoch:02d}_valacc{val_acc:.2f}_valloss{val_loss:.2f}.hdf5',
    #                               monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False,
    #                               mode='auto', period=1)

    reducelr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=0, mode='auto',
                                                 epsilon=0.0001, cooldown=0, min_lr=0)
    earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto')

    tensorboard = keras.callbacks.TensorBoard(log_dir='../logs/CNN_GRU', histogram_freq=0, write_graph=True,
                                              write_images=False,
                                              embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

    # csvlog = keras.callbacks.CSVLogger('../logs/log_CNN_GRU.csv')
    model = GRU_ATTENTION(input_size=word_size, output_size=y_num, input_length=input_length)
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(x_train_padded_seqs, y_train_one_hot,
              batch_size=64,
              epochs=200,
              validation_data=(x_test_padded_seqs, y_test_one_hot),
              callbacks=[check_point, reducelr, earlystop, tensorboard])


if __name__ == '__main__':
    train()
