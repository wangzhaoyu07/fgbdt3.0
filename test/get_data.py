import os
# import pandas as pd
# import numpy as np
# import pickle
os.chdir(os.path.join(__file__,os.path.pardir))

# # data=pd.read_csv('data_guest.csv',index_col=0)
# # print(data.head())
# # x=data[['imei','mac','cpucore']].values
# # y=np.zeros(len(x))
# # output_data={'X':x,'y':y}

# # output = open('data.pkl', 'wb')

# # pickle.dump(output_data, output)
# columns=[]
# for i in range(0,79):
#     columns.append('{}'.format(i))

# data=np.random.randn(75,79)
# data=pd.DataFrame(data,columns=columns)
# #print(data)
# y=np.zeros((len(data),1))
# tp=np.random.randint(0,50)
# for j in range(tp):
#     y[np.random.randint(0,74)]=1

# output_data={'X':data.values,'y':y}
# #print(data)
# output = open('testing-X.pkl', 'wb')

# pickle.dump(output_data, output)

# # for i in range(5):
# #     data=np.random.randn(75,79)
# #     data=pd.DataFrame(data,columns=columns)
# #     data['y']=np.zeros((len(data),1))
# #     for j in range(np.random.randint(0,50)):
# #         data['y'][np.random.randint(0,74)]=1
# #     data.to_csv('{}.csv'.format(i),index=False)




# import pandas as pd
# import numpy as np
# import pickle
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import SGDClassifier
# from sklearn.metrics import accuracy_score
# from sklearn import datasets
# import random

# data = datasets.load_iris()
# #print(data['data'])
# #print(data['target'])
# data=np.concatenate([data['data'],data['target'].reshape(-1,1)],axis=1)
# #print(data)
# df=pd.DataFrame(data,columns=['X0','X1','X2','X3','y'])
# #print(df.head())
# df=df[df['y']!=2]
# #print(df)

# #X_train,X_test,y_train,y_test=train_test_split(df.drop(['y'],axis=1),df['y'],test_size=0.2,random_state=0)

# #model=SGDClassifier().fit(X_train,y_train)
# #y_pred=model.predict(X_test)

# idx=list(np.arange(len(df)))
# random.shuffle(idx)
# #print(idx)
# df=df.iloc[idx]
# print(df)
# df.index=np.arange(len(df))

# #print(df)

# df_test=df[-20:]
# data_pkl={'X':df_test.drop(['y'],axis=1).values,'y':df_test['y'].values}
# print(df_test['y'].values)
# output = open('testing-X.pkl', 'wb')
# pickle.dump(data_pkl, output)

# df_train=df[:-10]
# for i in range(5):
#     tp=df[i*18:i*18+18]
#     tp.to_csv('../train/train/{}.csv'.format(i),index=False)

#大数据集
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn import datasets
import random
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier, train

df=pd.read_csv('train.csv')
#print(df.shape)
#print(df.head())
label=LabelEncoder().fit_transform(df['activity'])
df['activity']=label
df=df.drop(['rn'],axis=1)

X_train,X_test,y_train,y_test=train_test_split(df.drop(['activity'],axis=1),df['activity'],test_size=0.2,random_state=0)

# model = XGBClassifier(learning_rate=0.1,
#                       n_estimators=1000,         # 树的个数--1000棵树建立xgboost
#                       max_depth=8,               # 树的深度
#                       min_child_weight = 1,      # 叶子节点最小权重
#                       gamma=0.,                  # 惩罚项中叶子结点个数前的参e=0.8,       数
# #                       subsampl      # 随机选择80%样本建立决策树
#                       colsample_btree=0.8,       # 随机选择80%特征建立决策树
#                       objective='multi:softmax', # 指定损失函数
#                       scale_pos_weight=1,        # 解决样本个数不平衡的问题
#                       random_state=27            # 随机数
#                       )
# model.fit(X_train,
#           y_train,
#           eval_set = [(X_test,y_test)],
#           eval_metric = "mlogloss",
#           early_stopping_rounds = 10,
#           verbose = True)

# y_pred=model.predict(X_train)
# accuracy = accuracy_score(y_train,y_pred)
# print(accuracy) #1.0

# y_pred=model.predict(X_test)
# accuracy = accuracy_score(y_test,y_pred)
# print(accuracy) #0.9695

data_pkl={'X':X_test.values,'y':y_test.values}
#print(y_test)
output = open('testing-X.pkl', 'wb')
pickle.dump(data_pkl, output)

df_train=pd.concat([X_train,y_train],axis=1)
train_len=len(df_train)//5
for i in range(5):
    tp=df_train[i*train_len:i*train_len+train_len]
    tp.to_csv('../train/train/{}.csv'.format(i),index=False)