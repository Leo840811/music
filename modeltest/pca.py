import sys
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns

def pca(filename):

    data = pd.read_csv('./finalcsv/finalcsv30s.csv',index_col=0)
    data = data.dropna()
    data = data.drop(columns=['song_name','videoname','url']) # 刪除不要的欄位
    userdf_3s2 = pd.read_csv(f'./csv30s/{file_name}30s.csv', index_col=False)
    userdf_3s2 = userdf_3s2.dropna()
    userdf_3s2 = userdf_3s2.drop(columns=['Unnamed: 0','song_name']) # 刪除不要的欄位




    data = pd.concat([data, userdf_3s2],ignore_index=True, axis=0)

    data = data.groupby('songid').mean() # 以index 分群
    songid = data.index
    data['songid']=songid
    data
    data['label']=data['songid'].str.split("[0-9]").str.get(0)
    data = data.drop(columns=['songid'])
    data = data.reset_index()
    data = data.drop(columns=['songid'])
    data
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
    finalDf
    # pca.explained_variance_ratio_
    plt.figure(figsize = (16, 9))
    sns.scatterplot(x = "principal component 1", y = "principal component 2", data = finalDf, hue = "label", alpha = 0.7,
                   s = 100);

    plt.title('PCA on Genres', fontsize = 25)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 10);
    plt.xlabel("Principal Component 1", fontsize = 15)
    plt.ylabel("Principal Component 2", fontsize = 15)
    plt.savefig(f'{file_name}.jpg')
    print('save complete!')
    
    
    

# call function
file_name = sys.argv[1]
pca(file_name)