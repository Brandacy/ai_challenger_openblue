"""
-------------------------------------------------
   File Name:     predict_20model
   Description :
   Author :       weiwanshun
   date：          2018/9/14
-------------------------------------------------
   Change Activity:
                   2018/9/14:
-------------------------------------------------
"""
__author__ = 'weiwanshun'

key_list = [
    "location_traffic_convenience",
    "location_distance_from_business_district",
    "location_easy_to_find",
    "service_wait_time",
    "service_waiters_attitude",
    "service_parking_convenience",
    "service_serving_speed",
    "price_level",
    "price_cost_effective",
    "price_discount",
    "environment_decoration",
    "environment_noise",
    "environment_space",
    "environment_cleaness",
    "dish_portion",
    "dish_taste",
    "dish_look",
    "dish_recommendation",
    "others_overall_experience",
    "others_willing_to_consume_again"
]
from keras.models import load_model
import pandas as pd
from models_generator import AttLayer
from keras.preprocessing.sequence import pad_sequences
import json
from keras.preprocessing.text import Tokenizer
import numpy as np
import jieba

import os

import tensorflow as tf
from keras import backend as K

num_cores = 8
os.environ["CUDA_VISIBLE_DEVICES"] = ""

config = tf.ConfigProto(intra_op_parallelism_threads=num_cores, inter_op_parallelism_threads=num_cores,
                        allow_soft_placement=True, device_count={'CPU': num_cores})
session = tf.Session(config=config)
K.set_session(session)

#
# config = tf.ConfigProto()
# config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
# sess = tf.Session(config=config)
#
# KTF.set_session(sess)


df = pd.read_csv("../data/ai_challenger/ai_challenger_sentiment_analysis_validationset_20180816/sentiment_analysis_validationset.csv",
                 encoding="utf-8")

content = df["content"]
content_list = []
for text in content:
    token = jieba.cut(text)

    arr_temp = []
    for item in token:
        arr_temp.append(item)
    content_list.append(" ".join(arr_temp))

filters = '!?"#$%&()*+,-./:;<=>@[\\]^_`{|}~\t\n\r！@#￥%…&*（）：“”’‘；《》？，。'
tokenizer = Tokenizer(filters=filters, lower=True, split=" ", oov_token="UNK")
with open("vocab.json", encoding="utf-8") as f:
    vocab = json.load(f)
    tokenizer.word_index = vocab

content_list_seq = tokenizer.texts_to_sequences(content_list)

content_list_seq_pad = pad_sequences(content_list_seq, maxlen=300)

for folder in key_list:
    print("processing------------->", folder)
    file_list = os.listdir(folder)
    model_file = file_list[0]
    model = load_model("./" + folder + "/" + model_file, custom_objects={'AttLayer': AttLayer})
    results = model.predict(content_list_seq_pad)
    label_list = []
    for item in results:

        idx = np.argmax(item)
        if idx == 0:
            label_list.append(-2)
            continue
        if idx == 1:
            label_list.append(-1)
            continue
        if idx == 2:
            label_list.append(-0)
            continue
        if idx == 3:
            label_list.append(-1)

    df[folder] = label_list
    del model

df.to_csv("./validation_20label.csv", encoding="utf-8")