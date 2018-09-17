"""
-------------------------------------------------
   File Name：     prediction
   Description :
   Author :       weiwanshun
   date：          2018/9/11
-------------------------------------------------
   Change Activity:
                   2018/9/11:
-------------------------------------------------
"""
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

__author__ = 'weiwanshun'
intention_model = "./model_files/20180909104219##epoch02_valacc0.90_valloss0.26.hdf5"
model_segment_list = [
    "dish_look20180910092650##epoch02_valacc0.85_valloss0.36.hdf5",
    "dish_portion20180909191038##epoch02_valacc0.85_valloss0.34.hdf5",
    "dish_recommendation20180910105655##epoch03_valacc0.92_valloss0.23.hdf5",
    "dish_taste20180910001439##epoch02_valacc0.84_valloss0.34.hdf5",
    "environment_cleaness20180909145750##epoch02_valacc0.89_valloss0.27.hdf5",
    "environment_decoration20180909020855##epoch02_valacc0.89_valloss0.26.hdf5",
    "environment_noise20180909072907##epoch02_valacc0.89_valloss0.29.hdf5",
    "environment_space20180909104835##epoch02_valacc0.86_valloss0.35.hdf5",
    "location_distance_from_business_district20180907211433##epoch03_valacc0.97_valloss0.13.hdf5",
    "location_easy_to_find20180907235442##epoch03_valacc0.89_valloss0.33.hdf5",
    "location_traffic_convenience20180907183412##epoch02_valacc0.94_valloss0.19.hdf5",
    "others_overall_experience20180910142759##epoch01_valacc0.87_valloss0.30.hdf5",
    "others_willing_to_consume_again20180910194829##epoch02_valacc0.94_valloss0.17.hdf5",
    "price_cost_effective20180908192541##epoch03_valacc0.88_valloss0.30.hdf5",
    "price_discount20180908222520##epoch01_valacc0.83_valloss0.36.hdf5",
    "price_level20180908134112##epoch02_valacc0.82_valloss0.39.hdf5",
    "service_parking_convenience20180908105124##epoch02_valacc0.76_valloss0.53.hdf5",
    "service_serving_speed20180908113819##epoch03_valacc0.85_valloss0.38.hdf5",
    "service_waiters_attitude20180908040329##epoch02_valacc0.89_valloss0.26.hdf5",
    "service_wait_time20180908023013##epoch04_valacc0.72_valloss0.60.hdf5"
]

df = pd.read_csv("../data/ai_challenger/ai_challenger_sentiment_analysis_validationset_20180816/sentiment_analysis_validationset.csv",
                 encoding="utf-8")
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

model_segment_dic = {}

for model_name in model_segment_list:
    sub_list = model_name.split("2018")

    model_segment_dic[sub_list[0]] = model_name

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

content_list_seq_pad = pad_sequences(content_list_seq, maxlen=600)
print(content_list_seq_pad.shape)
intent_model = load_model(intention_model, custom_objects={'AttLayer': AttLayer})

results = intent_model.predict(content_list_seq_pad)
print(results.shape)

for i, key in enumerate(key_list):

    K.clear_session()
    tf.reset_default_graph()

    segment_model = load_model("./segment/" + model_segment_dic[key], custom_objects={'AttLayer': AttLayer})
    label_list = []
    j = 0
    for result in results:
        intent = result[i]
        segment = -2
        if intent > 0.4:
            sentence = content_list_seq_pad[j]
            sentence = np.array([sentence])
            # print(sentence.shape)
            segment_result = segment_model.predict(sentence)

            # print(segment_result.shape)
            max_segment = np.argmax(segment_result[0])
            if max_segment == 0:
                segment = 1
            elif max_segment == 1:
                segment = 0
            else:
                segment = -1

        label_list.append(segment)
        j += 1
    df[key] = label_list
df.to_csv("./validation.csv", encoding="utf-8")
