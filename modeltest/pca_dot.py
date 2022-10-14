import sys
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
import os
from os import listdir
from os.path import isfile, join

def pca(filename):


    parentpath = './pca_jpg/'
    pca_jpg_output = f'./pca_jpg/{filename}/'
    if not os.path.isdir(parentpath):
        os.mkdir(parentpath)
    if not os.path.isdir(pca_jpg_output):
        os.mkdir(pca_jpg_output)

    data = pd.read_csv('./finalcsv/finalcsv30s.csv',index_col=0)
    data = data.dropna()
    # data = data.drop(columns=['song_name','videoname','url'],axis=1) # 刪除不要的欄位
    userdf_3s2 = pd.read_csv(f'./csv30s/{file_name}30s.csv', index_col=False)
    userdf_3s2 = userdf_3s2.dropna()
    userdf_3s2 = userdf_3s2.drop(columns=['Unnamed: 0','song_name']) # 刪除不要的欄位

    data = pd.concat([data, userdf_3s2],ignore_index=True, axis=0)


    data = data.groupby('songid').mean() # 以index 分群

    songid = data.index
    data['songid']=songid
    data['label']=data['songid'].str.split("[0-9]").str.get(0)
    data = data.drop(columns=['songid'])
    data = data.reset_index()
    data = data.drop(columns=['songid'])

    data_poke = data.loc[data[data.label == file_name].index]
    data = data.drop(data[data.label == file_name].index)

    data = pd.concat([data_poke, data], ignore_index=True)

    y = data['label']
    X = data.loc[:, data.columns != 'label']

    # #### NORMALIZE X ####
    cols = X.columns
    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(X)
    X = pd.DataFrame(np_scaled, columns = cols)



    # #### PCA 2 COMPONENTS ####


    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(X)
    principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])


    # concatenate with target label
    finalDf = pd.concat([principalDf, y], axis = 1)

    # pca.explained_variance_ratio_
    plt.figure(figsize = (16, 9))
    sns.scatterplot(x = "principal component 1", y = "principal component 2", data = finalDf, hue = "label", alpha = 0.7, s= 20, size="label" , sizes={'blues':20,'pop':20, 'hiphop':20, 'classical':20, 'disco':20, 'country':20, 'rock':20, f'{file_name}':60, 'metal':20, 'jazz':20, 'reggae':20},\
        palette={'blues':'#f08080','pop':'#C2C287', 'hiphop':'#2894FF', 'classical':'#CC6600', 'disco':'#CC9933', 'country':'#009966', 'rock':'#FF77FF', f'{file_name}':'#AE0000', 'metal':'#DAB1D5', 'jazz':'#9F35FF', 'reggae':'#C4C400'});

    plt.title('PCA on Genres', fontsize = 25)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 10);
    plt.xlabel("Principal Component 1", fontsize = 15)
    plt.ylabel("Principal Component 2", fontsize = 15)
    plt.savefig(f'{pca_jpg_output}{file_name}.jpg')
    print('save complete!')

    # 轉成dataframe才能使用
    # df = pd.DataFrame(y,columns=['label'])

    # label_list = y.unique().tolist()
    # y = y.apply(lambda x : label_list.index(x))
    # lda = LDA(n_components=2)
    # X_r2 = lda.fit(X, y).transform(X)

    # plt.figure(figsize=(20, 10))
    # color_dict = {'blues':'#f08080','pop':'#C2C287', 'hiphop':'#2894FF', 'classical':'#CC6600', 'disco':'#CC9933', 'country':'#009966', 'rock':'#FF77FF', f'{file_name}':'#000000', 'metal':'#DAB1D5', 'jazz':'#9F35FF', 'reggae':'#C4C400'}
    # for  i, target_name, color_dict[target_name] in zip(range(0,522) ,label_list, color_dict):
    #     plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1],  label=target_name, c=color_dict[target_name])
    # plt.legend(loc='upper left')
    # plt.title('LDA of IRIS dataset')
    # plt.show()

    
    
    

# call function
file_name = sys.argv[1]
pca(file_name)