import pandas
from sklearn import preprocessing
import numpy as np


term1=0
term2=0
df1 = pandas.read_csv('/home/tongqing/test.csv',header=None)
df2 = pandas.read_csv('/home/tongqing/jer.csv',header=None)

acc=df1.values.tolist()   #将数据正转成列表
acc_=np.array(acc)   # 将列表转成数组
min_max_scaler = preprocessing.MinMaxScaler()
acc_n = min_max_scaler.fit_transform(acc_)     #将每列进行归一化

for i in range(len(df1)):
    a=acc_n[i]
    print(a)
    a=(np.linalg.norm(a))**2
    print(a)
    term1+=a
print("term1 ist:",term1)


jer=df2.values.tolist()   #将数据正转成列表
jer_=np.array(jer)   # 将列表转成数组
min_max_scaler = preprocessing.MinMaxScaler()
jer_n = min_max_scaler.fit_transform(jer_)

for i in range(len(df2)):
    j=jer_n[i]
    j=(np.linalg.norm(j))**2
    print(j)
    term2+=j
print("term2 ist:",term2)


penalty=term1+term2
print("smooth penalty ist:",penalty)












