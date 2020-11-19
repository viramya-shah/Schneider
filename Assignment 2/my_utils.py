import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math

from collections import Counter
from itertools import chain
from scipy.spatial.distance import cosine

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

def eucledian_distance(x, y):
    return math.sqrt(sum([(i-j)**2 for i, j in zip(x, y)]))


def iou(x, y):
    return sum(x & y)/sum(x)


def manhattan_distance(x, y):
    return sum([math.fabs(i-j)for i, j in zip(x, y)])

def read_data(file_path='./data/input/', file_name='netflix_titles.csv', index=None):
    return pd.read_csv(os.path.join(file_path, file_name))


def filter_with_type(df, type_value='Movie'):
    return df[df.type == type_value].reset_index(drop=True)


def clean_duration(df):
    return df.duration.apply(lambda x: int(x.split()[0]))


def removeNonAscii(s):
    return "".join(i for i in s if ord(i) < 128)


def clean_dataframe(df):
    # date_added
    if df.date_added.isna().any():
        # filling with the oldest date
        df.date_added.fillna('1-Jan-08', inplace=True)

    # title
    if df.title.isna().any():
        df.title.fillna('', inplace=True)  # filling with the empty string

    # director
    if df.director.isna().any():
        df.director.fillna('', inplace=True)  # filling with the empty string
    df['director'] = df.director.apply(removeNonAscii)

    # cast
    if df.cast.isna().any():
        df.cast.fillna('', inplace=True)  # filling with the empty string
    df['cast'] = df.cast.apply(removeNonAscii)

    # country
    if df.country.isna().any():
        df.country.fillna('', inplace=True)  # filling with the empty string

    # release_year
    if df.release_year.isna().any():
        # filling with the mean
        df.release_year.fillna(round(df.release_year.mean()), inplace=True)

    # rating
    if df.rating.isna().any():
        # filling with the most common rating
        df.rating.fillna('TV-MA', inplace=True)

    # duration
    if df.duration.isna().any():
        df.duration.fillna('0 min', inplace=True)  # filling with 0 mins

    # listed_in
    if df.listed_in.isna().any():
        df.listed_in.fillna([''], inplace=True)  # filling with empty string

    # description
    if df.description.isna().any():
        df.description.fillna('', inplace=True)  # filling with empty string
    df['director'] = df.director.apply(removeNonAscii)

    return df


def reformat_dataframe(df):
    # director
    df['director'] = df.director.apply(lambda x: x.split(", "))

    # cast
    df['cast'] = df.cast.apply(lambda x: x.split(", "))

    # country
    df['country'] = df.country.apply(lambda x: x.split(", "))

    # date_added
    df['date_added'] = pd.to_datetime(df.date_added)

    # listed_in
    df['listed_in'] = df.listed_in.apply(lambda x: x.split(", "))

    return df


def encode_dataframe(df):
    # rating
    df['rating'] = df.rating.astype('category')
    mapping = {i: idx for idx, i in enumerate(df.rating.cat.categories)}
    df['rating'] = df.rating.map(mapping)

    # duration
    df['duration'] = df.duration.apply(lambda x: int(x.split()[0]))
    return df


def save_csv(df, out_path="./data/output/", file_name='clean_netflix.csv'):
    df.to_csv(os.path.join(out_path, file_name))


def feature_engg(df):
    # COUNTRY
    country = []
    for i in df.country:
        for j in str(i)[1:-1].split(", "):
            country.append(j)
    country_d = Counter(country)
    print("There are {} different countries".format(len(country_d.keys())))

    # taking primary country
    df['country'] = df["country"].apply(eval)
    df['country_main'] = df.country.apply(lambda x: x[0])
    df['country_main'] = df.country_main.astype('category')
    df['country_main'] = df.country_main.cat.codes

    # number of countries
    num_country = []
    for c in df.country:
        if len(c[0]) == 0:
            num_country.append(0)
        else:
            num_country.append(len(c))
    df['num_country'] = num_country

    # DIRECTOR
    # director factor
    df['director'] = df.director.apply(eval)
    final_dict = {}
    director_dict = dict(Counter(list(chain.from_iterable(df.director))))
    factor = 1.0/sum(director_dict.values())
    normal_director_dict = {k: v*factor for k, v in director_dict.items()}

    # number of directors
    director_factor, num_d = [], []
    for directors in df.director:
        if len(directors[0]) == 0:
            director_factor.append(0)
            num_d.append(0)
        else:
            director_factor.append(
                sum([normal_director_dict[i] for i in directors])/len(directors))
            num_d.append(len(directors))
    df['director_factor'] = director_factor
    df['num_directores'] = num_d

    # CAST
    # cast factor
    df['cast'] = df.cast.apply(eval)
    final_dict_cast = {}
    cast_dict = dict(Counter(list(chain.from_iterable(df.cast))))
    cast_factor = 1.0/sum(cast_dict.values())
    normal_cast_dict = {k: v*cast_factor for k, v in cast_dict.items()}

    # number of cast
    # to add
    num_c, cast_factor_list = [], []
    for c in df.cast:
        if len(c[0]) == 0:
            num_c.append(0)
            cast_factor_list.append(0)
        else:
            num_c.append(len(c))
            cast_factor_list.append(
                sum([normal_cast_dict[i] for i in c])/len(c))
    df['cast_factor'] = cast_factor_list
    df['num_cast'] = num_c

    # GENRE
    genre = []
    for i in df.listed_in:
        for j in str(i)[1:-1].split(", "):
            genre.append(j)
    genre_d = Counter(genre)
    print("There are {} different genres".format(len(genre_d.keys())))

    df['listed_in'] = df.listed_in.apply(eval)

    # DATE
    df['date_added'] = pd.to_datetime(df.date_added)
    df['year'] = df.date_added.dt.year
    df['month'] = df.date_added.dt.month
    df['date'] = df.date_added.dt.day

    return df


def calculate_errors(test_label_df, pred, idxs = [], test_subsets = []):
    i1, i2, i3 = idxs[0], idxs[1], idxs[2]
    p1, p2, p3 = pred[i1], pred[i2], pred[i3]
    t1, t2, t3 = test_subsets[0], test_subsets[1], test_subsets[2]

    error_l2 = sum([eucledian_distance(i, j) for i, j in zip(test_label_df.values, pred)])/len(test_label_df)
    error_l2_1 = sum([eucledian_distance(i, j) for i, j in zip(t1, p1)])/len(t1)
    error_l2_2 = sum([eucledian_distance(i, j) for i, j in zip(t2, p2)])/len(t2)
    error_l2_3 = sum([eucledian_distance(i, j) for i, j in zip(t3, p3)])/len(t3)

    error_l1 = sum([manhattan_distance(i, j) for i, j in zip(test_label_df.values, pred)])/len(test_label_df)
    error_l1_1 = sum([manhattan_distance(i, j) for i, j in zip(t1, p1)])/len(t1)
    error_l1_2 = sum([manhattan_distance(i, j) for i, j in zip(t2, p2)])/len(t2)
    error_l1_3 = sum([manhattan_distance(i, j) for i, j in zip(t3, p3)])/len(t3)

    error_iou = sum([iou(i, j) for i, j in zip(test_label_df.values, pred)])/len(test_label_df)
    error_iou_1 = sum([iou(i, j) for i, j in zip(t1, p1)])/len(t1)
    error_iou_2 = sum([iou(i, j) for i, j in zip(t2, p2)])/len(t2)
    error_iou_3 = sum([iou(i, j) for i, j in zip(t3, p3)])/len(t3)

    error = {
        'Total L2 error': error_l2, 
        'Total L1 error': error_l1,
        'Total IoU error': error_iou,
        'L2 for 1 genre': error_l2_1,
        'L2 for 2 genre': error_l2_2,
        'L2 for 3 genre': error_l2_3,
        'L1 for 1 genre': error_l1_1,
        'L1 for 2 genre': error_l1_2,
        'L1 for 3 genre': error_l1_3,
        'IoU for 1 genre': error_iou_1,
        'IoU for 2 genre': error_iou_2,
        'IoU for 3 genre': error_iou_3,

    }

    return error
