import pickle
from PIL import Image
import PIL.ImageOps 
import numpy as np
from pathlib import Path
import gzip
import pandas as pd
from sklearn import preprocessing
from collections import Counter
import sys

def predict_ans(file_name):
    modeldir = './model'
    
    userdf_3s = pd.read_csv(f'./csv3s/{file_name}3s.csv', index_col=0)
    userdf_3s = userdf_3s.dropna()
    userdf_3s = userdf_3s.drop(['song_name','songid','label'], axis=1)

    # y = userdf_3s['label']
    X = userdf_3s.loc[:, userdf_3s.columns] # label以外的欄位

    # 對特徵值做標準化
    cols = X.columns
    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(X)

    # 新dataframe用標準化過的特徵值
    X = pd.DataFrame(np_scaled, columns = cols)

    # 讀取Model
#     with open(f'{modeldir}/xgbtest.pickle', 'rb') as xgbm1:
#         xgb1 = pickle.load(xgbm1)
#         pred = xgb1.predict(X)
#         df_pred = pd.DataFrame(pred)
#         df_pred = df_pred.replace([0,1,2,3,4,5,6,7,8,9],['blues','classical','country','disco','hiphop','jazz','metal','pop'
#         ,'reggae','rock'])
#         df_pred['曲風'] = df_pred[0]
#         predans = df_pred.groupby(['曲風']).size().reset_index(name='次數')
#         predans['機率'] = predans['次數']/ predans['次數'].sum()
#         predans = predans.sort_values(by='機率',ascending=False)
#         predans = predans.drop(['次數'], axis=1)        
# #        return predans
#         print(predans)
    # 讀取gzip.Model
    with gzip.open(f'{modeldir}/xgbtest.pgz', 'r') as xgbm2:
        xgb2 = pickle.load(xgbm2)
        pred = xgb2.predict(X)
        df_pred = pd.DataFrame(pred)
        df_pred = df_pred.replace([0,1,2,3,4,5,6,7,8,9],['blues','classical','country','disco','hiphop','jazz','metal','pop'
        ,'reggae','rock'])
        df_pred['曲風'] = df_pred[0]
        predans = df_pred.groupby(['曲風']).size().reset_index(name='次數')
        predans['機率'] = predans['次數']/ predans['次數'].sum()
        predans = predans.sort_values(by='機率',ascending=False)
        predans = predans.drop(['次數'], axis=1)
#        return predans
        print(predans)
    
file_name = sys.argv[1]    
predict_ans(file_name)