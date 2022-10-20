import sys
import librosa
import numpy as np
import time
import os, re
from os import listdir
from os.path import isfile, join
import keras
import tensorflow as tf
from keras.models import load_model
from keras.layers import Input

def load_DLmodel(file_name):
    id = 1  # Song ID
    X_user = np.empty((0, 128, 130))
    genres = []
    # directory_list = list()
    path = './musicfile/3s/%s/' %(file_name)

    file_data = [f for f in listdir(path) if isfile (join(path, f))]

    for line in file_data:
        if ( line[-1:] == '\n' ):   # 從換行前面取黨檔名
            line = line[:-1]

        songname = path + line # 將 目錄路徑跟檔名 合併成 檔案路徑
        y, sr = librosa.load(songname, sr=22050, duration=60) # 用 librosa讀取檔案
        S = np.abs(librosa.stft(y)) # 傅立葉轉換取振幅
        melspectrogram = librosa.feature.melspectrogram(S=S, sr=sr)
        melspectrogram = np.expand_dims(melspectrogram, axis = 0)
        X_user = np.append(X_user,melspectrogram, axis=0)
        id = id+1

    X_user = X_user.swapaxes(1,2)
    model = tf.keras.models.load_model('./model/crnnmodel.h5')
    n_features = X_user.shape[2]
    input_shape = (None, n_features)
    model_input = Input(input_shape, name='input')
    predict = model.predict(X_user).argmax(axis=1)
    dict_genres = {'blues':0, 'classical':1, 'country':2, 'disco':3, 
               'hiphop':4,'jazz':5, 'metal' :6, 'pop': 7 ,'reggae': 8 ,'rock':9}
    ans = np.argmax(np.bincount(predict))
    arr_genres = np.array(list(dict_genres.keys()))
    genre = arr_genres[ans]
    
    print(genre)


file_name = sys.argv[1]    
load_DLmodel(file_name)