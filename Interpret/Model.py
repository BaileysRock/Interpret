"""
用于存放机器学习模型
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn import metrics
from numpy.random import shuffle
from sklearn import svm
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn import metrics
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from CodingPorcessing import *
import pickle
import joblib


# sklearn实现的mlp
def MlpModel(X_train, Y_train, X_test, Y_test, choice):
    """
    训练mlp模型
    :param choice: choice为0，训练adult模型，choice为1，训练german模型
    """
    # 导入内置MLP模型
    # hidden_layer_sizes参数
    # 迭代次数设置为400
    # 三层神经网络
    if choice == 0:
        # adult的参数
        model = MLPClassifier(solver='sgd', activation='relu', alpha=1e-4, hidden_layer_sizes=(50, 50, 50),
                              random_state=1, max_iter=400, verbose=10, learning_rate_init=.1)
    elif choice == 1:
        # german的参数
        model = MLPClassifier(solver='sgd', activation='relu', alpha=1e-4, hidden_layer_sizes=(50, 50, 50),
                              random_state=1, max_iter=500, verbose=10, learning_rate_init=.1)
    else:
        model = None
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    print(confusion_matrix(Y_test, predictions))
    print(classification_report(Y_test, predictions))
    # print(mlp.score(X_test, Y_test))
    # print(mlp.n_layers_)
    # print(mlp.n_iter_)
    # print(mlp.loss_)
    # print(mlp.out_activation_)
    return model


# sklearn实现的svm
def SvmModel(X_train, Y_train, X_test, Y_test, choice):
    """
    :param choice: choice为0，训练adult模型，choice为1，训练german模型
    """
    if choice == 0:
        model = svm.SVC(C=2, kernel='rbf', gamma=10, decision_function_shape='ovo')
    elif choice == 1:
        model = svm.SVC(C=2, kernel='rbf', gamma=10, decision_function_shape='ovo')
    else:
        model = None
    model.fit(X_train, Y_train)
    train_score = model.score(X_train, Y_train)
    print("训练集：", train_score)
    test_score = model.score(X_train, Y_train)
    print("测试集：", test_score)
    pre_train = model.predict(X_train)  # 得到模型关于训练数据集的分类结果
    pre_test = model.predict(X_test)  # 得到模型关于测试数据集的分类结果
    print("Train Accuracy:%.4f\n" % metrics.accuracy_score(Y_train, pre_train))
    print("Test  Accuracy:%.4f\n" % metrics.accuracy_score(Y_test, pre_test))
    # model = svm.SVC(C=1.0, kernel='rbf', gamma='auto', decision_function_shape='ovr', cache_size=500)
    # model.fit(X_train, Y_train)
    return model


# sklearn实现的xgboost
def Xgboost(X_train, Y_train, X_test, Y_test, choice):
    """
    :param choice: choice为0，训练adult模型，choice为1，训练german模型
    """

    if choice == 0:
        model = XGBClassifier(n_estimators=5000, max_depth=10, min_child_weight=2,
                              gamma=0.9, subsample=0.8, colsample_bytree=0.8,
                              objective='binary:logitraw', nthread=-1,
                              scale_pos_weight=1)
        model_name = "./model/adult_xgboost.pkl"
    elif choice == 1:
        model = XGBClassifier(n_estimators=2000, max_depth=6, min_child_weight=2,
                              gamma=0.9, subsample=0.8, colsample_bytree=0.8,
                              objective='binary:logitraw', nthread=-1,
                              scale_pos_weight=1)
        model_name = "./model/german_xgboost.pkl"
    else:
        model = None
        model_name = ""
    model.fit(X_train, Y_train)
    Predictions = model.predict(X_test)
    scoreRandomForest = model.score(X_train, Y_train)
    print("randomforest prediction:")
    print(classification_report(Y_test, Predictions))
    print(scoreRandomForest)
    joblib.dump(filename=model_name, value=model)
    return model


# def XgboostClassifier(X_train,Y_train,X_test,Y_test):
#     xgbModel = XGBClassifier()
#     xgbModel.fit(X_train, Y_train)
#     print('The accuracy of eXtreme Gradient Boosting Classifier on testing set:', xgbModel.score(X_test, Y_test))
#     return xgbModel

# sklearn实现的randomforest
def randomForest(X_train, Y_train, X_test, Y_test, choice):
    """
    :param choice: choice为0，训练adult模型，choice为1，训练german模型
    """
    # random forest
    # 训练随机森林

    if choice == 0:
        randomForest = RandomForestClassifier(n_estimators=800, criterion='gini', max_depth=None, min_samples_split=2,
                                              min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto',
                                              max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True,
                                              oob_score=True, n_jobs=1, random_state=None, verbose=0,
                                              warm_start=False, class_weight=None)
        model_name = "./model/adult_randomforest.pkl"
    elif choice == 1:
        randomForest = RandomForestClassifier(n_estimators=2000, criterion='gini', max_depth=20, min_samples_split=2,
                                              min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto',
                                              max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True,
                                              oob_score=True, n_jobs=1, random_state=None, verbose=0,
                                              warm_start=False, class_weight=None)
        model_name = "./model/german_randomforest.pkl"
    else:
        randomForest = None
        model_name = ""
    randomForest.fit(X_train, Y_train)
    Predictions = randomForest.predict(X_test)
    scoreRandomForest = randomForest.score(X_train, Y_train)
    print("randomforest prediction:")
    print(classification_report(Y_test, Predictions))
    print(scoreRandomForest)
    joblib.dump(filename=model_name, value=randomForest)
    return randomForest


if __name__ == '__main__':
    adult_train_y, adult_train_X, adult_test_y, adult_test_X, german_train_y, german_train_X, german_test_y, german_test_X = loadData()

    # choice = 0、1、2，分别测试不同randomforest、xgboost、svm
    choice = 0
    if choice == 0:
        # adult的模型
        adult_randomforest = randomForest(X_train=adult_train_X, Y_train=adult_train_y, X_test=adult_test_X,
                                          Y_test=adult_test_y, choice=0)
        # german对应的模型
        german_randomforest = randomForest(X_train=german_train_X, Y_train=german_train_y, X_test=german_test_X,
                                           Y_test=german_test_y, choice=1)
    elif choice == 1:
        # adult的模型
        adult_xgboost = Xgboost(X_train=adult_train_X, Y_train=adult_train_y, X_test=adult_test_X,
                                Y_test=adult_test_y, choice=0)
        # german的模型
        german_xgboost = Xgboost(X_train=german_train_X, Y_train=german_train_y, X_test=german_test_X,
                                 Y_test=german_test_y, choice=1)
    else:
        pass
