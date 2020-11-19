import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from itertools import chain
from scipy.spatial.distance import cosine
import math
import pickle
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier


from my_utils import read_data, feature_engg, calculate_errors, eucledian_distance, manhattan_distance, iou


if __name__ == '__main__':
    FILE_PATH = './data/output'
    FILE_NAME = 'post_cluster.csv'

    df = read_data(FILE_PATH, FILE_NAME).iloc[:, 2:]

    ''' FEATURE ENGG'''
    df = feature_engg(df)

    '''CLASSIFICATION'''
    selected_cols = ['labels', 'year', 'month', 'date', 'country_main', 'num_country', 'director_factor',
                     'num_directores', 'num_cast', 'cast_factor', 'duration', 'rating', 'release_year']

    X_train, X_test, y_train, y_test = train_test_split(
        df[selected_cols],
        df.listed_in,
        test_size=0.2
    )

    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)

    label_encoder = MultiLabelBinarizer()
    train_label = label_encoder.fit_transform(y_train)
    train_label_df = pd.DataFrame(train_label, columns=label_encoder.classes_)

    test_label = label_encoder.transform(y_test)
    test_label_df = pd.DataFrame(test_label, columns=label_encoder.classes_)

    
    # saving data
    pickle.dump(X_train, open("./data/input/X_train.pkl", 'wb'))
    pickle.dump(X_test, open("./data/input/X_test.pkl", 'wb'))
    pickle.dump(train_label_df, open("./data/input/y_train.pkl", 'wb'))
    pickle.dump(test_label_df, open("./data/input/y_test.pkl", 'wb'))

    '''
        Bifurcating the results: with only 1 genre, 2 genres, and 3 genres
    '''

    idx_1 = test_label_df[test_label_df.sum(axis=1) == 1].index
    idx_2 = test_label_df[test_label_df.sum(axis=1) == 2].index
    idx_3 = test_label_df[test_label_df.sum(axis=1) == 3].index

    t1 = test_label_df[test_label_df.sum(axis=1) == 1].values
    t2 = test_label_df[test_label_df.sum(axis=1) == 2].values
    t3 = test_label_df[test_label_df.sum(axis=1) == 3].values

    
    # corelations
    fig, ax = plt.subplots(figsize=(16,10))         
    sns.heatmap(train_label_df.corr(), annot=True, linewidths=.5, ax=ax)
    ax.set_title("Heatmap shocasing correlations among Genres")
    ax.set_xlabel("Variables")
    ax.set_ylabel("Variables")
    plt.savefig("./data/output/corr_genres.png")

    fig, ax = plt.subplots(figsize=(16,10))         
    sns.heatmap(pd.concat([X_train, train_label_df], axis = 1).corr()[selected_cols], annot=True, linewidths=.5, ax=ax)
    ax.set_title("Heatmap shocasing correlations among variables")
    ax.set_xlabel("Variables")
    ax.set_ylabel("Variables")
    plt.savefig("./data/output/corr_variables.png")


    '''MODELING'''
    # RANDOM FOREST CLASSIFICATION
    parameters_rf = {
        'n_estimators': np.arange(10, 500, 50),
        'criterion': ['gini', 'entropy']
    }

    randomForest_gs = GridSearchCV(
        RandomForestClassifier(),
        parameters_rf,
        cv=3,
        refit=True,
        verbose=1,
        n_jobs=-1
    )

    randomForest_gs.fit(X_train, train_label_df)
    pred_rf = randomForest_gs.best_estimator_.predict(X_test)
    pickle.dump(randomForest_gs.best_estimator_, open("./data/output/models/rf.pkl", 'wb'))

    error_l2 = sum([eucledian_distance(i, j) for i, j in zip(
        test_label_df.values, pred_rf)])/len(test_label_df)
    error_l1 = sum([manhattan_distance(i, j) for i, j in zip(
        test_label_df.values, pred_rf)])/len(test_label_df)
    error_iou = sum([iou(i, j) for i, j in zip(
        test_label_df.values, pred_rf)])/len(test_label_df)

    error_rf = calculate_errors(test_label_df, pred_rf, [idx_1, idx_2, idx_3], [t1, t2, t3])
    error_rf_df = pd.DataFrame(error_rf, index = ['Random Forest'])

    # KNN ClASSIFICATON
    parameters_knn = {
        'n_neighbors': np.arange(1, 15, 2),
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    }

    knn_gs = GridSearchCV(
        KNeighborsClassifier(),
        parameters_knn,
        cv=3,
        refit=True,
        verbose=1,
        n_jobs=-1
    )

    knn_gs.fit(X_train, train_label_df)
    pred_knn = knn_gs.best_estimator_.predict(X_test)
    pickle.dump(knn_gs.best_estimator_, open("./data/output/models/knn.pkl", 'wb'))

    error_l2_knn = sum([eucledian_distance(i, j) for i, j in zip(
        test_label_df.values, pred_knn)])/len(test_label_df)
    error_l1_knn = sum([manhattan_distance(i, j) for i, j in zip(
        test_label_df.values, pred_knn)])/len(test_label_df)
    error_iou_knn = sum([iou(i, j) for i, j in zip(
        test_label_df.values, pred_knn)])/len(test_label_df)

    error_knn = calculate_errors(test_label_df, pred_knn, [idx_1, idx_2, idx_3], [t1, t2, t3])
    error_knn_df = pd.DataFrame(error_knn, index = ["KNN"])

    # MLP Classifier
    param_mlp = {
        'hidden_layer_sizes': [(64, 32, 16), (256, 128, 64, 32)],
        'activation': ['logistic', 'tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'learning_rate_init': [1e-4, 1e-3, 1e-2]
    }

    mlp_gs = GridSearchCV(
        MLPClassifier(max_iter = 1000), 
        param_mlp,
        cv = 3,
        refit = True,
        verbose = 1,
        n_jobs = -1
    )

    mlp_gs.fit(X_train, train_label_df)
    pickle.dump(mlp_gs.best_estimator_, open("./data/output/models/mlp.pkl", 'wb'))
    pred_mlp = mlp_gs.best_estimator_.predict(X_test)

    error_l2_mlp = sum([eucledian_distance(i, j) for i, j in zip(test_label_df.values, pred_mlp)])/len(test_label_df)
    error_l1_mlp = sum([manhattan_distance(i, j) for i, j in zip(test_label_df.values, pred_mlp)])/len(test_label_df)
    error_iou_mlp = sum([iou(i, j) for i, j in zip(test_label_df.values, pred_mlp)])/len(test_label_df)

    error_mlp = calculate_errors(test_label_df, pred_mlp, [idx_1, idx_2, idx_3], [t1, t2, t3])
    error_mlp_df = pd.DataFrame(error_mlp, index = ["MLP"])

    # Extra tree classifier
    parameters_etc = {
        'n_estimators': np.arange(10, 500, 50),
        'criterion': ['gini', 'entropy']
    }

    extratree_gs = GridSearchCV(
        ExtraTreesClassifier(),
        parameters_etc, 
        cv = 3,
        refit = True,
        verbose = 1,
        n_jobs = -1
    )

    extratree_gs.fit(X_train, train_label_df)
    pred_etc = extratree_gs.best_estimator_.predict(X_test)
    pickle.dump(extratree_gs.best_estimator_, open("./data/output/models/etc.pkl", 'wb'))

    error_l2_etc = sum([eucledian_distance(i, j) for i, j in zip(test_label_df.values, pred_etc)])/len(test_label_df)
    error_l1_etc = sum([manhattan_distance(i, j) for i, j in zip(test_label_df.values, pred_etc)])/len(test_label_df)
    error_iou_etc = sum([iou(i, j) for i, j in zip(test_label_df.values, pred_etc)])/len(test_label_df)

    error_etc = calculate_errors(test_label_df, pred_etc, [idx_1, idx_2, idx_3], [t1, t2, t3])
    error_etc_df = pd.DataFrame(error_etc, index = ['Extra Trees'])

    final_result = pd.concat([
        error_rf_df, error_knn_df, error_mlp_df, error_etc_df
    ])

    final_result.to_csv("./data/output/final_result.csv")