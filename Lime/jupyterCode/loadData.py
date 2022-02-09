from sklearn.preprocessing import LabelEncoder
import pandas as pd


def load_data():
    """
    读取数据并返回
    :return:
    """
    adultTrain = pd.read_csv("../../DataSetProcess/adultTrain.txt")
    adultTest = pd.read_csv("../../DataSetProcess/adultTest.txt")
    germanTrain = pd.read_csv("../../DataSetProcess/germanTrain.txt")
    germanTest = pd.read_csv("../../DataSetProcess/germanTest.txt")
    labelEncoderAdultTrain = LabelEncoder()
    adultTrain['income'] = labelEncoderAdultTrain.fit_transform(adultTrain['income'])
    labelEncoderAdultTest = LabelEncoder()
    adultTest['income'] = labelEncoderAdultTest.fit_transform(adultTest['income'])
    labelEncoderGermanTrain = LabelEncoder()
    germanTrain['class'] = labelEncoderGermanTrain.fit_transform(germanTrain['class'])
    labelEncoderGermanTest = LabelEncoder()
    germanTest['class'] = labelEncoderGermanTest.fit_transform(germanTest['class'])
    adult_train_y = adultTrain["income"]
    adult_train_X = adultTrain.drop('income', axis=1)
    adult_test_y = adultTest["income"]
    adult_test_X = adultTest.drop('income', axis=1)
    german_train_y = germanTrain["class"]
    german_train_X = germanTrain.drop('class', axis=1)
    german_test_y = germanTest["class"]
    german_test_X = germanTest.drop('class', axis=1)

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

    return adult_train_y, adult_train_X, adult_test_y, adult_test_X, german_train_y, german_train_X, german_test_y, german_test_X, Adult, German
