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

from keras.callbacks import ModelCheckpoint
from data_preprocess import *
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


def train(model_path, x, y, input_length, word_size, y_num):
    print("train ===============> ", model_path)
    x_train_padded_seqs, x_test_padded_seqs, y_train_one_hot, y_test_one_hot = train_test_split(x,
                                                                                                y,
                                                                                                test_size=0.15,
                                                                                                random_state=666)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    check_point = ModelCheckpoint(filepath=model_path + time.strftime("%Y%m%d%H%M%S",
                                                                      time.localtime()) + '##epoch{epoch:02d}_valacc{val_acc:.2f}_valloss{val_loss:.2f}.hdf5',
                                  monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False,
                                  mode='auto', period=1)

    reducelr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=0, mode='auto',
                                                 epsilon=0.0001, cooldown=0, min_lr=0)
    earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, verbose=1, mode='auto')

    tensorboard = keras.callbacks.TensorBoard(log_dir='../logs/CNN_GRU', histogram_freq=0, write_graph=True,
                                              write_images=False,
                                              embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

    # csvlog = keras.callbacks.CSVLogger('../logs/log_CNN_GRU.csv')
    model = GRU_ATTENTION(input_size=word_size, output_size=y_num, input_length=input_length)
    model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(x_train_padded_seqs, y_train_one_hot,
              batch_size=64,
              epochs=50,
              validation_data=(x_test_padded_seqs, y_test_one_hot),
              callbacks=[check_point, reducelr, earlystop, tensorboard])


if __name__ == '__main__':

    data_dic, word_size, y_num = get_segment_train_data(600)
    for key in data_dic.keys():
        [x_str, y] = data_dic[key]

        x = []
        for item in x_str:

            chars = str(item).split(",")
            sub_x = []
            for c in chars:
                sub_x.append(int(c))

            x.append(sub_x)
        x = np.array(x)
        y = np.array(y)
        train(key, x, y, 600, word_size, y_num)
