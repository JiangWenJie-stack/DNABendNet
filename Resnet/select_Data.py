import csv
from pandas.core.frame import DataFrame
import pandas as pd
import numpy as np
import random

train_data = open("D:\JWJ\Huangcongcong\Data\\train_data.txt", "w")
vaild_data = open("D:\JWJ\Huangcongcong\Data\\vaild_data.txt", "w")
test_data = open("D:\JWJ\Huangcongcong\Data\\test_data.txt", "w")


data = pd.read_csv('D:\JWJ\Huangcongcong\Data\\no_repeat_alldata.csv')

row = len(data)
all_index = np.array(range(1,row+1))
test_index = random.sample(range(1, row), row//10)  ###在区间（1，row）中生成row*10%测试集的索引
vaild_index = random.sample(list(set(all_index) - set(test_index)), row//5)  ###在区间（1，row）中排除测试集后生成row*10%验证集的索引

for i in range(0,row):
    if i %999 == 0:
       print(i)
    if data.iat[i,0] in test_index:
        test_data.write(str(data.iloc[i]['seq']) + '\t' + str(data.iloc[i]['c0']) + '\n')
    elif data.iat[i,0] in vaild_index:
        vaild_data.write(str(data.iloc[i]['seq']) + '\t' + str(data.iloc[i]['c0']) + '\n')
    else:
        train_data.write(str(data.iloc[i]['seq']) + '\t' + str(data.iloc[i]['c0']) + '\n')

test_data.close()
vaild_data.close()
train_data.close()


