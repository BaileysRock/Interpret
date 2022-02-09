"""
用于解释模型
"""
import joblib
from matplotlib import pyplot as plt
from pdpbox import pdp
import xgboost as xgb
import shap
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from CodingPorcessing import loadData


def pdpPlot(model, X_test, data, dir: str):
    """
    画pdp的图
    :param model: 待解释的模型
    :param X_test: X_test部分的数据
    :param data: 该数据集的所有数据
    :param dir: adult/german + Randomforest/Xgboost 用于制图,eg:adultRandomforest
    """
    columns = data.columns
    # 获取所有特征，并保存到list
    datacolumns = []
    for this_column in columns:
        if this_column != "predict":
            datacolumns.append(this_column)
    # 跑所有特征的pdp图
    for feature in datacolumns:
        pdp_feature = pdp.pdp_isolate(model=model, dataset=X_test, model_features=datacolumns, feature=feature)
        # plot it
        pdp.pdp_plot(pdp_feature, feature)
        # plt.savefig("C:/Users/Alienware/Desktop/plot/pdp/xgboost_german/"+feature+".png")
        plt.savefig("./Plot/Pdp/" + dir + "/" + feature + ".png")
        plt.show()


def shapPlot(model, X_train, data, dir: str):
    """
    计算shap values并跑出shap的图
    :param model: 模型
    :param X_train: 训练数据
    :param data: 所有的数据
    :param dir: adult/german + Randomforest/Xgboost 用于写入数据和制图,eg:adultRandomforest
    """
    explainer = shap.TreeExplainer(model)  # 模型训练用什么矩阵形状，这里要对应
    shap_values = explainer.shap_values(X_train)  # 传入特征矩阵X，计算SHAP值
    columns = data.columns
    positivePath = './DataShap/' + dir + '_positive.csv'
    negativePath = './DataShap/' + dir + '_negative.csv'
    np.savetxt(positivePath, shap_values[0], fmt='%f')
    np.savetxt(negativePath, shap_values[1], fmt='%f')
    datacolumns = []
    for this_column in columns:
        if this_column != "predict":
            datacolumns.append(this_column)
    shap.summary_plot(shap_values, datacolumns, plot_type="bar", show=False)
    # shap.summary_plot(shap_values, datacolumns)
    # plt.show()
    # plt.figure(figsize=(500, 500))
    picturePath = "./Plot/Shap/" + dir + ".png"
    plt.savefig(picturePath, bbox_inches='tight')
    # plt.show()
    # shap.summary_plot(shap_values, datacolumns, plot_type="bar")


if __name__ == '__main__':

    # 读取数据
    adult_train_y, adult_train_X, adult_test_y, adult_test_X, german_train_y, german_train_X, german_test_y, german_test_X = loadData()
    # 处理数据
    # 获得测试集数据
    AdultTrain = pd.concat([adult_train_X, adult_train_y], axis=1)
    GermanTrain = pd.concat([german_train_X, german_train_y], axis=1)
    # 获得训练集数据
    AdultTest = pd.concat([adult_test_X, adult_test_y], axis=1)
    GermanTest = pd.concat([german_test_X, german_test_y], axis=1)
    # 获得完整数据集
    Adult = pd.concat([AdultTrain, AdultTest], axis=0)
    German = pd.concat([GermanTrain, GermanTest], axis=0)

    # 读取adult的模型
    adult_randomforest_model = joblib.load("./model/adult_randomforest.pkl")
    adult_xgboost_model = joblib.load("./model/adult_xgboost.pkl")
    # 读取german的模型
    german_randomforest_model = joblib.load("./model/german_randomforest.pkl")
    german_xgboost_model = joblib.load("./model/german_xgboost.pkl")

    choice = 0
    # choice = 0 使用shap解释
    # choice = 1 使用pdp解释
    if choice == 0:
        # randomforest
        shapPlot(model=german_randomforest_model, X_train=german_train_X, data=German, dir="germanRandomforest")
        # shapPlot(model=adult_randomforest_model, X_train=adult_train_X, data=Adult, dir="adultRandomforest")

        shapPlot(model=german_xgboost_model,X_train=german_train_X, data=German, dir="germanXgboost")
        shapPlot(model=adult_xgboost_model,X_train=adult_train_X,data = Adult,dir="adultXgboost")
    else:
        pass
