"""
用于处理数据，将字符串用数字表示出来
"""

from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

def loadAdult():
    """
    读取数据，并处理缺失值
    :return: 处理后的数据
    """
    filePathAdult = "./DataSet/adult.csv"
    # 读取文件
    data_adult = pd.read_csv(filePathAdult)
    # 处理缺失值
    features = [feature for feature in data_adult.columns]
    for feature in features:
        data_adult.loc[data_adult[feature] == ' ?', feature] = data_adult[feature].mode()[0]
    return data_adult

def loadGerman():
    """
    读取数据，并处理缺失值
    :return: 处理后的数据
    """
    filePathGerman = "./DataSet/german.csv"
    # 读取文件
    data_german = pd.read_csv(filePathGerman)
    # 处理缺失值
    features = [feature for feature in data_german.columns]
    for feature in features:
        data_german.loc[data_german[feature] == ' ?', feature] = data_german[feature].mode()[0]
    return data_german


def loadData():
    """
    读取数据并返回
    :return:
    """
    adultTrain = pd.read_csv("./DataSetProcess/adultTrain.txt")
    adultTest = pd.read_csv("./DataSetProcess/adultTest.txt")
    germanTrain = pd.read_csv("./DataSetProcess/germanTrain.txt")
    germanTest = pd.read_csv("./DataSetProcess/germanTest.txt")
    labelEncoderAdultTrain = LabelEncoder()
    adultTrain['income'] = labelEncoderAdultTrain.fit_transform(adultTrain['income'])
    labelEncoderAdultTest = LabelEncoder()
    adultTest['income'] = labelEncoderAdultTest.fit_transform(adultTest['income'])
    labelEncoderGermanTrain = LabelEncoder()
    germanTrain['class'] = labelEncoderGermanTrain.fit_transform(germanTrain['class'])
    labelEncoderGermanTest = LabelEncoder()
    germanTest['class'] = labelEncoderGermanTest.fit_transform(germanTest['class'])
    if (labelEncoderAdultTrain.classes_ == labelEncoderAdultTest.classes_).all() == False:
        exit()
    if (labelEncoderGermanTrain.classes_ == labelEncoderGermanTest.classes_).all() == False:
        exit()
    print(labelEncoderAdultTrain.classes_)
    print(labelEncoderGermanTrain.classes_)
    adult_train_y = adultTrain["income"]
    adult_train_X = adultTrain.drop('income', axis=1)
    adult_test_y = adultTest["income"]
    adult_test_X = adultTest.drop('income', axis=1)
    german_train_y = germanTrain["class"]
    german_train_X = germanTrain.drop('class', axis=1)
    german_test_y = germanTest["class"]
    german_test_X = germanTest.drop('class', axis=1)
    return adult_train_y, adult_train_X, adult_test_y, adult_test_X, german_train_y, german_train_X, german_test_y, german_test_X

if __name__ == "__main__":
    # dataAdult = loadAdult()
    # dataGerman = loadGerman()

    # 读取处理缺失值后的数据
    dataAdult = pd.read_csv("./DataSetPre/adult.csv")
    dataGerman = pd.read_csv("./DataSetPre/german.csv")

    dataAdultX = dataAdult.drop('income', axis=1)
    dataAdultY = dataAdult['income']

    dataGermanX = dataGerman.drop('class',axis=1)
    dataGermanY = dataGerman['class']

    # 对特征读热编码
    dataAdultX = pd.get_dummies(dataAdultX)
    dataGermanX = pd.get_dummies(dataGermanX)

    # 读热后拼接数据集
    dataAdult = pd.concat([dataAdultX, dataAdultY], axis=1)
    dataGerman = pd.concat([dataGermanX,dataGermanY],axis=1)

    # 分割数据集
    adultTrain,adultTest = train_test_split(dataAdult)
    germanTrain,germanTest = train_test_split(dataGerman)

    # 分别将数据写入文件
    adultTrain.to_csv("./DataSetProcess/adultTrain.txt",index=False)
    adultTest.to_csv("./DataSetProcess/adultTest.txt",index=False)

    germanTrain.to_csv("./DataSetProcess/germanTrain.txt",index=False)
    germanTest.to_csv("./DataSetProcess/germanTest.txt",index=False)



