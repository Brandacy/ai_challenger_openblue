"""
-------------------------------------------------
   File Name：     data_preprocess
   Description :
   Author :       weiwanshun
   date：          2018/9/6
-------------------------------------------------
   Change Activity:
                   2018/9/6:
-------------------------------------------------
"""
import pandas as pd
import numpy as np
import jieba
import os

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
import json
from keras.preprocessing.text import Tokenizer

__author__ = 'weiwanshun'


def get_train_data(input_length,
                   path="../data/ai_challenger/ai_challenger_sentiment_analysis_trainingset_20180816/sentiment_analysis_trainingset.csv"):
    df = pd.read_csv(path, encoding="utf-8")
    # id = df["id"]
    content = df["content"]
    location_traffic_convenience = df["location_traffic_convenience"]
    location_distance_from_business_district = df["location_distance_from_business_district"]
    location_easy_to_find = df["location_easy_to_find"]
    service_wait_time = df["service_wait_time"]
    service_waiters_attitude = df["service_waiters_attitude"]
    service_parking_convenience = df["service_parking_convenience"]
    service_serving_speed = df["service_serving_speed"]
    price_level = df["price_level"]
    price_cost_effective = df["price_cost_effective"]
    price_discount = df["price_discount"]
    environment_decoration = df["environment_decoration"]
    environment_noise = df["environment_noise"]
    environment_space = df["environment_space"]
    environment_cleaness = df["environment_cleaness"]
    dish_portion = df["dish_portion"]
    dish_taste = df["dish_taste"]
    dish_look = df["dish_look"]
    dish_recommendation = df["dish_recommendation"]
    others_overall_experience = df["others_overall_experience"]
    others_willing_to_consume_again = df["others_willing_to_consume_again"]

    label_list = []
    for i, review in enumerate(content):
        label = [
            location_traffic_convenience[i],
            location_distance_from_business_district[i],
            location_easy_to_find[i],
            service_wait_time[i],
            service_waiters_attitude[i],
            service_parking_convenience[i],
            service_serving_speed[i],
            price_level[i],
            price_cost_effective[i],
            price_discount[i],
            environment_decoration[i],
            environment_noise[i],
            environment_space[i],
            environment_cleaness[i],
            dish_portion[i],
            dish_taste[i],
            dish_look[i],
            dish_recommendation[i],
            others_overall_experience[i],
            others_willing_to_consume_again[i]
        ]
        for idx in range(len(label)):
            if label[idx] == -2:
                label[idx] = 0
            else:
                label[idx] = 1

        label_list.append(label)

    label_list = np.array(label_list)
    content_list = []
    for text in content:
        token = jieba.cut(text)

        arr_temp = []
        for item in token:
            arr_temp.append(item)
        content_list.append(" ".join(arr_temp))
    filters = '!?"#$%&()*+,-./:;<=>@[\\]^_`{|}~\t\n\r！@#￥%…&*（）：“”’‘；《》？，。'
    tokenizer = Tokenizer(filters=filters, lower=True, split=" ", oov_token="UNK")

    if os.path.exists("vocab.json"):
        with open("vocab.json", encoding="utf-8") as f:

            vocab = json.load(f)
            tokenizer.word_index = vocab
    else:
        tokenizer.fit_on_texts(content_list)
        vocab = tokenizer.word_index
        with open("vocab.json", encoding="utf-8", mode="w") as f:

            json.dump(vocab, f)

    content_list_seq = tokenizer.texts_to_sequences(content_list)

    content_list_seq_pad = pad_sequences(content_list_seq, maxlen=input_length)

    return content_list_seq_pad, label_list, len(vocab) + 1, len(label)


def get_segment_train_data(input_length,
                           path="train_data_after_cut.xlsx"):
    df = pd.read_excel(path, encoding="utf-8")
    # id = df[df["id"]!= -2]
    content = df["content"]
    # content_list = []
    # for text in content:
    #     token = jieba.cut(text)
    #
    #     arr_temp = []
    #     for item in token:
    #         arr_temp.append(item)
    #     content_list.append(" ".join(arr_temp))
    filters = '!?"#$%&()*+,-./:;<=>@[\\]^_`{|}~\t\n\r！@#￥%…&*（）：“”’‘；《》？，。'
    tokenizer = Tokenizer(filters=filters, lower=True, split=" ", oov_token="UNK")
    if os.path.exists("vocab.json"):
        with open("vocab.json", encoding="utf-8") as f:

            vocab = json.load(f)
            tokenizer.word_index = vocab
    else:
        tokenizer.fit_on_texts(content)
        vocab = tokenizer.word_index
        with open("vocab.json", encoding="utf-8", mode="w") as f:

            json.dump(vocab, f)

    content_list_seq = tokenizer.texts_to_sequences(content)
    # print(sum([len(c) for c in content_list_seq])/len(content_list_seq))
    content_list_seq_pad = pad_sequences(content_list_seq, maxlen=input_length)
    temp = []
    for content_l in content_list_seq_pad:
        temp.append(",".join(list(content_l.astype(str))))

    # print(content_list_seq_pad[0])
    df["content"] = temp
    # df.to_excel("train_data_sequences_pad.xlsx")

    location_traffic_convenience = df[df["location_traffic_convenience"] != -2]
    location_distance_from_business_district = df[df["location_distance_from_business_district"] != -2]
    location_easy_to_find = df[df["location_easy_to_find"] != -2]
    service_wait_time = df[df["service_wait_time"] != -2]
    service_waiters_attitude = df[df["service_waiters_attitude"] != -2]
    service_parking_convenience = df[df["service_parking_convenience"] != -2]
    service_serving_speed = df[df["service_serving_speed"] != -2]
    price_level = df[df["price_level"] != -2]
    price_cost_effective = df[df["price_cost_effective"] != -2]
    price_discount = df[df["price_discount"] != -2]
    environment_decoration = df[df["environment_decoration"] != -2]
    environment_noise = df[df["environment_noise"] != -2]
    environment_space = df[df["environment_space"] != -2]
    environment_cleaness = df[df["environment_cleaness"] != -2]
    dish_portion = df[df["dish_portion"] != -2]
    dish_taste = df[df["dish_taste"] != -2]
    dish_look = df[df["dish_look"] != -2]
    dish_recommendation = df[df["dish_recommendation"] != -2]
    others_overall_experience = df[df["others_overall_experience"] != -2]
    others_willing_to_consume_again = df[df["others_willing_to_consume_again"] != -2]

    location_traffic_convenience_label = location_traffic_convenience["location_traffic_convenience"]
    location_distance_from_business_district_label = location_distance_from_business_district[
        "location_distance_from_business_district"]
    location_easy_to_find_label = location_easy_to_find["location_easy_to_find"]
    service_wait_time_label = service_wait_time["service_wait_time"]
    service_waiters_attitude_label = service_waiters_attitude["service_waiters_attitude"]
    service_parking_convenience_label = service_parking_convenience["service_parking_convenience"]
    service_serving_speed_label = service_serving_speed["service_serving_speed"]
    price_level_label = price_level["price_level"]
    price_cost_effective_label = price_cost_effective["price_cost_effective"]
    price_discount_label = price_discount["price_discount"]
    environment_decoration_label = environment_decoration["environment_decoration"]
    environment_noise_label = environment_noise["environment_noise"]
    environment_space_label = environment_space["environment_space"]
    environment_cleaness_label = environment_cleaness["environment_cleaness"]
    dish_portion_label = dish_portion["dish_portion"]
    dish_taste_label = dish_taste["dish_taste"]
    dish_look_label = dish_look["dish_look"]
    dish_recommendation_label = dish_recommendation["dish_recommendation"]
    others_overall_experience_label = others_overall_experience["others_overall_experience"]
    others_willing_to_consume_again_label = others_willing_to_consume_again["others_willing_to_consume_again"]

    location_traffic_convenience_content = location_traffic_convenience["content"]
    location_distance_from_business_district_content = location_distance_from_business_district["content"]
    location_easy_to_find_content = location_easy_to_find["content"]
    service_wait_time_content = service_wait_time["content"]
    service_waiters_attitude_content = service_waiters_attitude["content"]
    service_parking_convenience_content = service_parking_convenience["content"]
    service_serving_speed_content = service_serving_speed["content"]
    price_level_content = price_level["content"]
    price_cost_effective_content = price_cost_effective["content"]
    price_discount_content = price_discount["content"]
    environment_decoration_content = environment_decoration["content"]
    environment_noise_content = environment_noise["content"]
    environment_space_content = environment_space["content"]
    environment_cleaness_content = environment_cleaness["content"]
    dish_portion_content = dish_portion["content"]
    dish_taste_content = dish_taste["content"]
    dish_look_content = dish_look["content"]
    dish_recommendation_content = dish_recommendation["content"]
    others_overall_experience_content = others_overall_experience["content"]
    others_willing_to_consume_again_content = others_willing_to_consume_again["content"]
    segment_data_dic = {
        "location_traffic_convenience": [location_traffic_convenience_content, location_traffic_convenience_label],
        "location_distance_from_business_district": [location_distance_from_business_district_content,
                                                     location_distance_from_business_district_label],
        "location_easy_to_find": [location_easy_to_find_content, location_easy_to_find_label],
        "service_wait_time": [service_wait_time_content, service_wait_time_label],
        "service_waiters_attitude": [service_waiters_attitude_content, service_waiters_attitude_label],
        "service_parking_convenience": [service_parking_convenience_content, service_parking_convenience_label],
        "service_serving_speed": [service_serving_speed_content, service_serving_speed_label],
        "price_level": [price_level_content, price_level_label],
        "price_cost_effective": [price_cost_effective_content, price_cost_effective_label],
        "price_discount": [price_discount_content, price_discount_label],
        "environment_decoration": [environment_decoration_content, environment_decoration_label],
        "environment_noise": [environment_noise_content, environment_noise_label],
        "environment_space": [environment_space_content, environment_space_label],
        "environment_cleaness": [environment_cleaness_content, environment_cleaness_label],
        "dish_portion": [dish_portion_content, dish_portion_label],
        "dish_taste": [dish_taste_content, dish_taste_label],
        "dish_look": [dish_look_content, dish_look_label],
        "dish_recommendation": [dish_recommendation_content, dish_recommendation_label],
        "others_overall_experience": [others_overall_experience_content, others_overall_experience_label],
        "others_willing_to_consume_again": [others_willing_to_consume_again_content,
                                            others_willing_to_consume_again_label]
    }
    for key in segment_data_dic.keys():

        label_list = segment_data_dic[key][1]

        label_list_onehot = []
        for data in label_list:

            label = [0, 0, 0]

            if data == 1:
                label[0] = 1
            if data == 0:
                label[1] = 1
            if data == -1:
                label[2] = 1
            label_list_onehot.append(label)
        segment_data_dic[key][1] = label_list_onehot

    return segment_data_dic, len(vocab), 3

    # get_segment_train_data(600)
def get_multilabel_train_data(input_length,
                           path="train_data_after_cut.xlsx"):
    df = pd.read_excel(path, encoding="utf-8")
    content = df["content"]
    filters = '!?"#$%&()*+,-./:;<=>@[\\]^_`{|}~\t\n\r！@#￥%…&*（）：“”’‘；《》？，。'
    tokenizer = Tokenizer(filters=filters, lower=True, split=" ", oov_token="UNK")
    if os.path.exists("vocab.json"):
        with open("vocab.json", encoding="utf-8") as f:

            vocab = json.load(f)
            tokenizer.word_index = vocab
    else:
        tokenizer.fit_on_texts(content)
        vocab = tokenizer.word_index
        with open("vocab.json", encoding="utf-8", mode="w") as f:

            json.dump(vocab, f)

    content_list_seq = tokenizer.texts_to_sequences(content)
    # print(sum([len(c) for c in content_list_seq])/len(content_list_seq))
    content_list_seq_pad = pad_sequences(content_list_seq, maxlen=input_length)

    return df,content_list_seq_pad,4,len(vocab)