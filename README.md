# 步骤

# 1.运行CodingProcessing获取预处理后的文件
由于有现成数据，该部分直接使用处理后的数据  
此部分功能:将数据读入，并处理缺失值，并将文件写入DataSetPre  
将数据分为测试集和训练集并保存

# 2.获得数据处理后的数据
将数据分为训练集和测试集，此处在CodingProcessing.py中有loaddata函数可以将数据读入  

# 3.使用loaddata函数获取adult、german的训练集和测试集

# 4.训练xgboost、randomfoerst模型，并保存到model中


# 5.在InterpretModel中选择解释


# 保存lime结果
jupyter nbconvert --to html limeInterpret.ipynb

