"""
-------------------------------------------------
   File Name：     score
   Description :
   Author :       weiwanshun
   date：          2018/9/12
-------------------------------------------------
   Change Activity:
                   2018/9/12:
-------------------------------------------------
"""
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

__author__ = 'weiwanshun'


def presion_and_recall(y_predict, y_label, class_type):
    size = len(y_label)

    TP = 0
    FN = 0
    FP = 0

    for i in range(size):

        if y_label[i] == class_type and y_predict[i] == class_type:
            TP += 1
            continue
        if y_label[i] == class_type:
            FN += 1
            continue
        if y_predict[i] == class_type:
            FP += 1

    presion = TP / (TP + FP)
    recall = TP / (TP + FN)

    return presion, recall


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


def f1_macro():
    df = pd.read_csv("validation.csv")
    df_val = pd.read_csv(
        "../data/ai_challenger/ai_challenger_sentiment_analysis_validationset_20180816/sentiment_analysis_validationset.csv")

    f1_sum = 0

    for cloumn in key_list:
        f1 = f1_score(df_val[cloumn], df[cloumn], average='macro')

        f1_sum += f1

    print(f1_sum / len(key_list))

f1_macro()