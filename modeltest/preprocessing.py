from pytube import Playlist
from moviepy.editor import *
from pydub import AudioSegment
from pydub.utils import make_chunks
from pydub import AudioSegment
import os, re
import random
import math
import pandas as pd
import librosa
import numpy as np
from os import listdir
from os.path import isfile, join
import sys

def preprocessing(file_name):
    
    mp4_dir = './musicfile/mp4/%s' %(file_name)   # 給檔案來源路徑
    mp4_parentfolder = './musicfile/wav'        # 給檔案輸出目錄的上一層目錄
    mp4_outputpath = './musicfile/wav/%s' %(file_name)   # 給檔案輸出目錄

    # 如果沒有資料夾建立資料夾
    if not os.path.isdir(mp4_parentfolder):
        os.mkdir(mp4_parentfolder)
    if not os.path.isdir(mp4_outputpath):
        os.mkdir(mp4_outputpath)

    initial_count = 1  # 為了下面range()取到檔案總數+1，才有取到全部

    for path in os.listdir(mp4_dir):   # 找出目錄路徑內的所有資料夾及檔案
        if os.path.isfile(os.path.join(mp4_dir, path)):    # 如果是檔案就抓出來
            initial_count += 1              # 計算總共有幾個檔案+1

    for n in range(1,initial_count):    # 從第一個到總數的檔案，一個一個轉檔
    # for n in range(1,2):
    # for n in os.listdir(dir):
        # 給mp4檔案來源名稱和路徑
        mp4_file = mp4_dir+'/%s.mp4' %(file_name)
        # 給wav檔案輸出名稱和路徑
        wav_file = mp4_outputpath+'/%s.wav' %(file_name)
        
        videoclip = VideoFileClip(mp4_file)   # 先匯入影檔
        audioclip = videoclip.audio           # 從影檔取出音檔
    #   audioclip.nchannels = 1    # 設定聲道數
        audioclip.write_audiofile(wav_file, fps=44000)  # 將取出的音檔寫入剛給的wav名稱及路經，並設採樣率44000
        audioclip.close()
        videoclip.close()    

    # wav分割成30s
    wav_dir = './musicfile/wav/%s' %(file_name)
    wav_parentfolder = './musicfile/30s'
    wav_outputpath = './musicfile/30s/%s' %(file_name)

    # 如果沒有資料夾建立資料夾
    if not os.path.isdir(wav_parentfolder):
        os.mkdir(wav_parentfolder)
    if not os.path.isdir(wav_outputpath):
        os.mkdir(wav_outputpath)

    # 迴圈目錄下所有檔案
    for each in os.listdir(wav_dir):
        
        filename = re.findall(r"(.*?)\.wav", each) # 取出.wav字尾的檔名
        if each:
            # filename[0] += '.wav'
            # print(filename[0])

            wav = AudioSegment.from_file((wav_dir+'/{}').format(each), "wav") # 開啟wav檔案
            size = 30000  # 切割的毫秒數 1s=1000

            chunks = make_chunks(wav, size)  # 將檔案切割為30s一塊
            numstop = len(chunks)-2
            for i, chunk in enumerate(chunks):    # 一段一段的chunk取出來
                if numstop>=i>=1:   # 不取前30s及最後30s的部分
#                 if i>=2 and len(chunk) >= 29999:   # 不取前60s及最後不到30s的部分
                    chunk_name = "{}-{}.wav".format(each.split(".")[0],i)
                    chunk.export((wav_outputpath+'/{}').format(chunk_name), format="wav")
                    
    # 30s分割成3s 
    dir_30s = './musicfile/30s/%s' %(file_name)
    parentfolder_30s = './musicfile/3s'
    outputpath_30s = './musicfile/3s/%s' %(file_name)

    # # 如果沒有資料夾建立資料夾
    if not os.path.isdir(parentfolder_30s):
        os.mkdir(parentfolder_30s)
    if not os.path.isdir(outputpath_30s):
        os.mkdir(outputpath_30s)

        # 迴圈目錄下所有檔案
    for each in os.listdir(dir_30s):

        filename = re.findall(r"(.*?)\.wav", each) # 取出.wav字尾的檔名
        if each:
            # filename[0] += '.wav'
            # print(filename[0])

            wav = AudioSegment.from_file((dir_30s+'/{}').format(each), "wav") # 開啟wav檔案
            size = 3000  # 切割的毫秒數 1s=1000
            chunks = make_chunks(wav, size)  # 將檔案切割為30s一塊

            for i, chunk in enumerate(chunks):
                if len(chunk)>=2999:
                    chunk_name = "{}-{}.wav".format(each.split(".")[0],i)
                    chunk.export((outputpath_30s+'/{}').format(chunk_name), format="wav")
    # 特徵提取30s
    path = './musicfile/30s/%s/' %(file_name)
    id = 1  # Song ID
    
    # 給一個 DataFrame表格變數
    feature_set3s30s = pd.DataFrame()  
    
    # 給個別的特徵一個 Series欄位變數
    songname_vector = pd.Series(dtype='float64')
    tempo_vector = pd.Series(dtype='float64')
    total_beats = pd.Series(dtype='float64')
    average_beats = pd.Series(dtype='float64')
    chroma_stft_mean = pd.Series(dtype='float64')
    chroma_stft_std = pd.Series(dtype='float64')
    chroma_stft_var = pd.Series(dtype='float64')
    chroma_cq_mean = pd.Series(dtype='float64')
    chroma_cq_std = pd.Series(dtype='float64')
    chroma_cq_var = pd.Series(dtype='float64')
    chroma_cens_mean = pd.Series(dtype='float64')
    chroma_cens_std = pd.Series(dtype='float64')
    chroma_cens_var = pd.Series(dtype='float64')
    mel_mean = pd.Series(dtype='float64')
    mel_std = pd.Series(dtype='float64')
    mel_var = pd.Series(dtype='float64')
    
#     mfcc_mean = pd.Series()
#     mfcc_std = pd.Series()
#     mfcc_var = pd.Series()
#     mfcc_delta_mean = pd.Series()
#     mfcc_delta_std = pd.Series()
#     mfcc_delta_var = pd.Series()

    mfcc1_mean = pd.Series(dtype='float64')
    mfcc2_mean = pd.Series(dtype='float64')
    mfcc3_mean = pd.Series(dtype='float64')
    mfcc4_mean = pd.Series(dtype='float64')
    mfcc5_mean = pd.Series(dtype='float64')
    mfcc6_mean = pd.Series(dtype='float64')
    mfcc7_mean = pd.Series(dtype='float64')
    mfcc8_mean = pd.Series(dtype='float64')
    mfcc9_mean = pd.Series(dtype='float64')
    mfcc10_mean = pd.Series(dtype='float64')
    mfcc11_mean = pd.Series(dtype='float64')
    mfcc12_mean = pd.Series(dtype='float64')
    mfcc13_mean = pd.Series(dtype='float64')
    mfcc14_mean = pd.Series(dtype='float64')
    mfcc15_mean = pd.Series(dtype='float64')
    mfcc16_mean = pd.Series(dtype='float64')
    mfcc17_mean = pd.Series(dtype='float64')
    mfcc18_mean = pd.Series(dtype='float64')
    mfcc19_mean = pd.Series(dtype='float64')
    mfcc20_mean = pd.Series(dtype='float64')
    
    mfcc1_std = pd.Series(dtype='float64')
    mfcc2_std = pd.Series(dtype='float64')
    mfcc3_std = pd.Series(dtype='float64')
    mfcc4_std = pd.Series(dtype='float64')
    mfcc5_std = pd.Series(dtype='float64')
    mfcc6_std = pd.Series(dtype='float64')
    mfcc7_std = pd.Series(dtype='float64')
    mfcc8_std = pd.Series(dtype='float64')
    mfcc9_std = pd.Series(dtype='float64')
    mfcc10_std = pd.Series(dtype='float64')
    mfcc11_std = pd.Series(dtype='float64')
    mfcc12_std = pd.Series(dtype='float64')
    mfcc13_std = pd.Series(dtype='float64')
    mfcc14_std = pd.Series(dtype='float64')
    mfcc15_std = pd.Series(dtype='float64')
    mfcc16_std = pd.Series(dtype='float64')
    mfcc17_std = pd.Series(dtype='float64')
    mfcc18_std = pd.Series(dtype='float64')
    mfcc19_std = pd.Series(dtype='float64')
    mfcc20_std = pd.Series(dtype='float64')

    mfcc1_var = pd.Series(dtype='float64')
    mfcc2_var = pd.Series(dtype='float64')
    mfcc3_var = pd.Series(dtype='float64')
    mfcc4_var = pd.Series(dtype='float64')
    mfcc5_var = pd.Series(dtype='float64')
    mfcc6_var = pd.Series(dtype='float64')
    mfcc7_var = pd.Series(dtype='float64')
    mfcc8_var = pd.Series(dtype='float64')
    mfcc9_var = pd.Series(dtype='float64')
    mfcc10_var = pd.Series(dtype='float64')
    mfcc11_var = pd.Series(dtype='float64')
    mfcc12_var = pd.Series(dtype='float64')
    mfcc13_var = pd.Series(dtype='float64')
    mfcc14_var = pd.Series(dtype='float64')
    mfcc15_var = pd.Series(dtype='float64')
    mfcc16_var = pd.Series(dtype='float64')
    mfcc17_var = pd.Series(dtype='float64')
    mfcc18_var = pd.Series(dtype='float64')
    mfcc19_var = pd.Series(dtype='float64')
    mfcc20_var = pd.Series(dtype='float64')

    mfcc1_delta_mean = pd.Series(dtype='float64')
    mfcc2_delta_mean = pd.Series(dtype='float64')
    mfcc3_delta_mean = pd.Series(dtype='float64')
    mfcc4_delta_mean = pd.Series(dtype='float64')
    mfcc5_delta_mean = pd.Series(dtype='float64')
    mfcc6_delta_mean = pd.Series(dtype='float64')
    mfcc7_delta_mean = pd.Series(dtype='float64')
    mfcc8_delta_mean = pd.Series(dtype='float64')
    mfcc9_delta_mean = pd.Series(dtype='float64')
    mfcc10_delta_mean = pd.Series(dtype='float64')
    mfcc11_delta_mean = pd.Series(dtype='float64')
    mfcc12_delta_mean = pd.Series(dtype='float64')
    mfcc13_delta_mean = pd.Series(dtype='float64')
    mfcc14_delta_mean = pd.Series(dtype='float64')
    mfcc15_delta_mean = pd.Series(dtype='float64')
    mfcc16_delta_mean = pd.Series(dtype='float64')
    mfcc17_delta_mean = pd.Series(dtype='float64')
    mfcc18_delta_mean = pd.Series(dtype='float64')
    mfcc19_delta_mean = pd.Series(dtype='float64')
    mfcc20_delta_mean = pd.Series(dtype='float64')
    
    mfcc1_delta_std = pd.Series(dtype='float64')
    mfcc2_delta_std = pd.Series(dtype='float64')
    mfcc3_delta_std = pd.Series(dtype='float64')
    mfcc4_delta_std = pd.Series(dtype='float64')
    mfcc5_delta_std = pd.Series(dtype='float64')
    mfcc6_delta_std = pd.Series(dtype='float64')
    mfcc7_delta_std = pd.Series(dtype='float64')
    mfcc8_delta_std = pd.Series(dtype='float64')
    mfcc9_delta_std = pd.Series(dtype='float64')
    mfcc10_delta_std = pd.Series(dtype='float64')
    mfcc11_delta_std = pd.Series(dtype='float64')
    mfcc12_delta_std = pd.Series(dtype='float64')
    mfcc13_delta_std = pd.Series(dtype='float64')
    mfcc14_delta_std = pd.Series(dtype='float64')
    mfcc15_delta_std = pd.Series(dtype='float64')
    mfcc16_delta_std = pd.Series(dtype='float64')
    mfcc17_delta_std = pd.Series(dtype='float64')
    mfcc18_delta_std = pd.Series(dtype='float64')
    mfcc19_delta_std = pd.Series(dtype='float64')
    mfcc20_delta_std = pd.Series(dtype='float64')

    mfcc1_delta_var = pd.Series(dtype='float64')
    mfcc2_delta_var = pd.Series(dtype='float64')
    mfcc3_delta_var = pd.Series(dtype='float64')
    mfcc4_delta_var = pd.Series(dtype='float64')
    mfcc5_delta_var = pd.Series(dtype='float64')
    mfcc6_delta_var = pd.Series(dtype='float64')
    mfcc7_delta_var = pd.Series(dtype='float64')
    mfcc8_delta_var = pd.Series(dtype='float64')
    mfcc9_delta_var = pd.Series(dtype='float64')
    mfcc10_delta_var = pd.Series(dtype='float64')
    mfcc11_delta_var = pd.Series(dtype='float64')
    mfcc12_delta_var = pd.Series(dtype='float64')
    mfcc13_delta_var = pd.Series(dtype='float64')
    mfcc14_delta_var = pd.Series(dtype='float64')
    mfcc15_delta_var = pd.Series(dtype='float64')
    mfcc16_delta_var = pd.Series(dtype='float64')
    mfcc17_delta_var = pd.Series(dtype='float64')
    mfcc18_delta_var = pd.Series(dtype='float64')
    mfcc19_delta_var = pd.Series(dtype='float64')
    mfcc20_delta_var = pd.Series(dtype='float64')

    rmse_mean = pd.Series(dtype='float64')
    rmse_std = pd.Series(dtype='float64')
    rmse_var = pd.Series(dtype='float64')
    cent_mean = pd.Series(dtype='float64')
    cent_std = pd.Series(dtype='float64')
    cent_var = pd.Series(dtype='float64')
    spec_bw_mean = pd.Series(dtype='float64')
    spec_bw_std = pd.Series(dtype='float64')
    spec_bw_var = pd.Series(dtype='float64')
    contrast_mean = pd.Series(dtype='float64')
    contrast_std = pd.Series(dtype='float64')
    contrast_var = pd.Series(dtype='float64')
    rolloff_mean = pd.Series(dtype='float64')
    rolloff_std = pd.Series(dtype='float64')
    rolloff_var = pd.Series(dtype='float64')
    poly_mean = pd.Series(dtype='float64')
    poly_std = pd.Series(dtype='float64')
    poly_var = pd.Series(dtype='float64')
    tonnetz_mean = pd.Series(dtype='float64')
    tonnetz_std = pd.Series(dtype='float64')
    tonnetz_var = pd.Series(dtype='float64')
    zcr_mean = pd.Series(dtype='float64')
    zcr_std = pd.Series(dtype='float64')
    zcr_var = pd.Series(dtype='float64')
    harm_mean = pd.Series(dtype='float64')
    harm_std = pd.Series(dtype='float64')
    harm_var = pd.Series(dtype='float64')
    perc_mean = pd.Series(dtype='float64')
    perc_std = pd.Series(dtype='float64')
    perc_var = pd.Series(dtype='float64')
    frame_mean = pd.Series(dtype='float64')
    frame_std = pd.Series(dtype='float64')
    frame_var = pd.Series(dtype='float64')
    
    
    # 找出路徑資料夾下的所有檔案 ,輸出成 list
    file_data = [f for f in listdir(path) if isfile (join(path, f))]
    
    # list內的每個元素(資料長相:'檔名')取出來
    for line in file_data:
        if ( line[-1:] == '\n' ):   # 從換行前面取黨檔名
            line = line[:-1]     # 不取逗號

        # 讀取音檔
        songname = path + line # 將 目錄路徑跟檔名 合併成 檔案路徑
        y, sr = librosa.load(songname) # 用 librosa讀取檔案
        S = np.abs(librosa.stft(y)) # 傅立葉轉換取振幅
        
        # 特徵提取
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
        melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        rmse = librosa.feature.rms(y=y)
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        poly_features = librosa.feature.poly_features(S=S, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        harmonic = librosa.effects.harmonic(y)
        percussive = librosa.effects.percussive(y)
        
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        mfcc_delta = librosa.feature.delta(mfcc)
    
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        frames_to_time = librosa.frames_to_time(onset_frames[:20], sr=sr)
        
        # 將提取出各個特徵的值丟進前面建的 Series欄位變數
        songname_vector._set_value(id, line)  # song name
        tempo_vector._set_value(id, tempo)  # tempo
        total_beats._set_value(id, sum(beats))  # beats
        average_beats._set_value(id, np.average(beats))
        chroma_stft_mean._set_value(id, np.mean(chroma_stft))  # chroma stft
        chroma_stft_std._set_value(id, np.std(chroma_stft))
        chroma_stft_var._set_value(id, np.var(chroma_stft))
        chroma_cq_mean._set_value(id, np.mean(chroma_cq))  # chroma cq
        chroma_cq_std._set_value(id, np.std(chroma_cq))
        chroma_cq_var._set_value(id, np.var(chroma_cq))
        chroma_cens_mean._set_value(id, np.mean(chroma_cens))  # chroma cens
        chroma_cens_std._set_value(id, np.std(chroma_cens))
        chroma_cens_var._set_value(id, np.var(chroma_cens))
        mel_mean._set_value(id, np.mean(melspectrogram))  # melspectrogram
        mel_std._set_value(id, np.std(melspectrogram))
        mel_var._set_value(id, np.var(melspectrogram))
        
#         mfcc_mean._set_value(id, np.mean(mfcc))
#         mfcc_std._set_value(id, np.std(mfcc))
#         mfcc_var._set_value(id, np.var(mfcc))
#         mfcc_delta_mean._set_value(id, np.mean(mfcc_delta))  # mfcc delta
#         mfcc_delta_std._set_value(id, np.std(mfcc_delta))
#         mfcc_delta_var._set_value(id, np.var(mfcc_delta))

        mfcc1_mean._set_value(id, np.mean(mfcc[0]))
        mfcc2_mean._set_value(id, np.mean(mfcc[1]))
        mfcc3_mean._set_value(id, np.mean(mfcc[2]))
        mfcc4_mean._set_value(id, np.mean(mfcc[3]))
        mfcc5_mean._set_value(id, np.mean(mfcc[4]))
        mfcc6_mean._set_value(id, np.mean(mfcc[5]))
        mfcc7_mean._set_value(id, np.mean(mfcc[6]))
        mfcc8_mean._set_value(id, np.mean(mfcc[7]))
        mfcc9_mean._set_value(id, np.mean(mfcc[8]))
        mfcc10_mean._set_value(id, np.mean(mfcc[9]))
        mfcc11_mean._set_value(id, np.mean(mfcc[10]))
        mfcc12_mean._set_value(id, np.mean(mfcc[11]))
        mfcc13_mean._set_value(id, np.mean(mfcc[12]))
        mfcc14_mean._set_value(id, np.mean(mfcc[13]))
        mfcc15_mean._set_value(id, np.mean(mfcc[14]))
        mfcc16_mean._set_value(id, np.mean(mfcc[15]))
        mfcc17_mean._set_value(id, np.mean(mfcc[16]))
        mfcc18_mean._set_value(id, np.mean(mfcc[17]))
        mfcc19_mean._set_value(id, np.mean(mfcc[18]))
        mfcc20_mean._set_value(id, np.mean(mfcc[19]))
        
        mfcc1_std._set_value(id, np.std(mfcc[0]))
        mfcc2_std._set_value(id, np.std(mfcc[1]))
        mfcc3_std._set_value(id, np.std(mfcc[2]))
        mfcc4_std._set_value(id, np.std(mfcc[3]))
        mfcc5_std._set_value(id, np.std(mfcc[4]))
        mfcc6_std._set_value(id, np.std(mfcc[5]))
        mfcc7_std._set_value(id, np.std(mfcc[6]))
        mfcc8_std._set_value(id, np.std(mfcc[7]))
        mfcc9_std._set_value(id, np.std(mfcc[8]))
        mfcc10_std._set_value(id, np.std(mfcc[9]))
        mfcc11_std._set_value(id, np.std(mfcc[10]))
        mfcc12_std._set_value(id, np.std(mfcc[11]))
        mfcc13_std._set_value(id, np.std(mfcc[12]))
        mfcc14_std._set_value(id, np.std(mfcc[13]))
        mfcc15_std._set_value(id, np.std(mfcc[14]))
        mfcc16_std._set_value(id, np.std(mfcc[15]))
        mfcc17_std._set_value(id, np.std(mfcc[16]))
        mfcc18_std._set_value(id, np.std(mfcc[17]))
        mfcc19_std._set_value(id, np.std(mfcc[18]))
        mfcc20_std._set_value(id, np.std(mfcc[19]))
        
        mfcc1_var._set_value(id, np.var(mfcc[0]))
        mfcc2_var._set_value(id, np.var(mfcc[1]))
        mfcc3_var._set_value(id, np.var(mfcc[2]))
        mfcc4_var._set_value(id, np.var(mfcc[3]))
        mfcc5_var._set_value(id, np.var(mfcc[4]))
        mfcc6_var._set_value(id, np.var(mfcc[5]))
        mfcc7_var._set_value(id, np.var(mfcc[6]))
        mfcc8_var._set_value(id, np.var(mfcc[7]))
        mfcc9_var._set_value(id, np.var(mfcc[8]))
        mfcc10_var._set_value(id, np.var(mfcc[9]))
        mfcc11_var._set_value(id, np.var(mfcc[10]))
        mfcc12_var._set_value(id, np.var(mfcc[11]))
        mfcc13_var._set_value(id, np.var(mfcc[12]))
        mfcc14_var._set_value(id, np.var(mfcc[13]))
        mfcc15_var._set_value(id, np.var(mfcc[14]))
        mfcc16_var._set_value(id, np.var(mfcc[15]))
        mfcc17_var._set_value(id, np.var(mfcc[16]))
        mfcc18_var._set_value(id, np.var(mfcc[17]))
        mfcc19_var._set_value(id, np.var(mfcc[18]))
        mfcc20_var._set_value(id, np.var(mfcc[19]))
        
        mfcc1_delta_mean._set_value(id, np.mean(mfcc_delta[0]))
        mfcc2_delta_mean._set_value(id, np.mean(mfcc_delta[1]))
        mfcc3_delta_mean._set_value(id, np.mean(mfcc_delta[2]))
        mfcc4_delta_mean._set_value(id, np.mean(mfcc_delta[3]))
        mfcc5_delta_mean._set_value(id, np.mean(mfcc_delta[4]))
        mfcc6_delta_mean._set_value(id, np.mean(mfcc_delta[5]))
        mfcc7_delta_mean._set_value(id, np.mean(mfcc_delta[6]))
        mfcc8_delta_mean._set_value(id, np.mean(mfcc_delta[7]))
        mfcc9_delta_mean._set_value(id, np.mean(mfcc_delta[8]))
        mfcc10_delta_mean._set_value(id, np.mean(mfcc_delta[9]))
        mfcc11_delta_mean._set_value(id, np.mean(mfcc_delta[10]))
        mfcc12_delta_mean._set_value(id, np.mean(mfcc_delta[11]))
        mfcc13_delta_mean._set_value(id, np.mean(mfcc_delta[12]))
        mfcc14_delta_mean._set_value(id, np.mean(mfcc_delta[13]))
        mfcc15_delta_mean._set_value(id, np.mean(mfcc_delta[14]))
        mfcc16_delta_mean._set_value(id, np.mean(mfcc_delta[15]))
        mfcc17_delta_mean._set_value(id, np.mean(mfcc_delta[16]))
        mfcc18_delta_mean._set_value(id, np.mean(mfcc_delta[17]))
        mfcc19_delta_mean._set_value(id, np.mean(mfcc_delta[18]))
        mfcc20_delta_mean._set_value(id, np.mean(mfcc_delta[19]))
        
        mfcc1_delta_std._set_value(id, np.std(mfcc_delta[0]))
        mfcc2_delta_std._set_value(id, np.std(mfcc_delta[1]))
        mfcc3_delta_std._set_value(id, np.std(mfcc_delta[2]))
        mfcc4_delta_std._set_value(id, np.std(mfcc_delta[3]))
        mfcc5_delta_std._set_value(id, np.std(mfcc_delta[4]))
        mfcc6_delta_std._set_value(id, np.std(mfcc_delta[5]))
        mfcc7_delta_std._set_value(id, np.std(mfcc_delta[6]))
        mfcc8_delta_std._set_value(id, np.std(mfcc_delta[7]))
        mfcc9_delta_std._set_value(id, np.std(mfcc_delta[8]))
        mfcc10_delta_std._set_value(id, np.std(mfcc_delta[9]))
        mfcc11_delta_std._set_value(id, np.std(mfcc_delta[10]))
        mfcc12_delta_std._set_value(id, np.std(mfcc_delta[11]))
        mfcc13_delta_std._set_value(id, np.std(mfcc_delta[12]))
        mfcc14_delta_std._set_value(id, np.std(mfcc_delta[13]))
        mfcc15_delta_std._set_value(id, np.std(mfcc_delta[14]))
        mfcc16_delta_std._set_value(id, np.std(mfcc_delta[15]))
        mfcc17_delta_std._set_value(id, np.std(mfcc_delta[16]))
        mfcc18_delta_std._set_value(id, np.std(mfcc_delta[17]))
        mfcc19_delta_std._set_value(id, np.std(mfcc_delta[18]))
        mfcc20_delta_std._set_value(id, np.std(mfcc_delta[19]))
        
        mfcc1_delta_var._set_value(id, np.var(mfcc_delta[0]))
        mfcc2_delta_var._set_value(id, np.var(mfcc_delta[1]))
        mfcc3_delta_var._set_value(id, np.var(mfcc_delta[2]))
        mfcc4_delta_var._set_value(id, np.var(mfcc_delta[3]))
        mfcc5_delta_var._set_value(id, np.var(mfcc_delta[4]))
        mfcc6_delta_var._set_value(id, np.var(mfcc_delta[5]))
        mfcc7_delta_var._set_value(id, np.var(mfcc_delta[6]))
        mfcc8_delta_var._set_value(id, np.var(mfcc_delta[7]))
        mfcc9_delta_var._set_value(id, np.var(mfcc_delta[8]))
        mfcc10_delta_var._set_value(id, np.var(mfcc_delta[9]))
        mfcc11_delta_var._set_value(id, np.var(mfcc_delta[10]))
        mfcc12_delta_var._set_value(id, np.var(mfcc_delta[11]))
        mfcc13_delta_var._set_value(id, np.var(mfcc_delta[12]))
        mfcc14_delta_var._set_value(id, np.var(mfcc_delta[13]))
        mfcc15_delta_var._set_value(id, np.var(mfcc_delta[14]))
        mfcc16_delta_var._set_value(id, np.var(mfcc_delta[15]))
        mfcc17_delta_var._set_value(id, np.var(mfcc_delta[16]))
        mfcc18_delta_var._set_value(id, np.var(mfcc_delta[17]))
        mfcc19_delta_var._set_value(id, np.var(mfcc_delta[18]))
        mfcc20_delta_var._set_value(id, np.var(mfcc_delta[19]))
        
        rmse_mean._set_value(id, np.mean(rmse))  # rmse
        rmse_std._set_value(id, np.std(rmse))
        rmse_var._set_value(id, np.var(rmse))
        cent_mean._set_value(id, np.mean(cent))  # cent
        cent_std._set_value(id, np.std(cent))
        cent_var._set_value(id, np.var(cent))
        spec_bw_mean._set_value(id, np.mean(spec_bw))  # spectral bandwidth
        spec_bw_std._set_value(id, np.std(spec_bw))
        spec_bw_var._set_value(id, np.var(spec_bw))
        contrast_mean._set_value(id, np.mean(contrast))  # contrast
        contrast_std._set_value(id, np.std(contrast))
        contrast_var._set_value(id, np.var(contrast))
        rolloff_mean._set_value(id, np.mean(rolloff))  # rolloff
        rolloff_std._set_value(id, np.std(rolloff))
        rolloff_var._set_value(id, np.var(rolloff))
        poly_mean._set_value(id, np.mean(poly_features))  # poly features
        poly_std._set_value(id, np.std(poly_features))
        poly_var._set_value(id, np.var(poly_features))
        tonnetz_mean._set_value(id, np.mean(tonnetz))  # tonnetz
        tonnetz_std._set_value(id, np.std(tonnetz))
        tonnetz_var._set_value(id, np.var(tonnetz))
        zcr_mean._set_value(id, np.mean(zcr))  # zero crossing rate
        zcr_std._set_value(id, np.std(zcr))
        zcr_var._set_value(id, np.var(zcr))
        harm_mean._set_value(id, np.mean(harmonic))  # harmonic
        harm_std._set_value(id, np.std(harmonic))
        harm_var._set_value(id, np.var(harmonic))
        perc_mean._set_value(id, np.mean(percussive))  # percussive
        perc_std._set_value(id, np.std(percussive))
        perc_var._set_value(id, np.var(percussive))
        frame_mean._set_value(id, np.mean(frames_to_time))  # frames
        frame_std._set_value(id, np.std(frames_to_time))
        frame_var._set_value(id, np.var(frames_to_time))
        
        
        id = id+1   #完成換下一首歌
    
    # 將各個 Series 加入前面建的 feature_set3s30s dataframe裡
    feature_set3s30s['song_name'] = songname_vector  # song name
    feature_set3s30s['tempo'] = tempo_vector  # tempo 
    feature_set3s30s['total_beats'] = total_beats  # beats
    feature_set3s30s['average_beats'] = average_beats
    feature_set3s30s['chroma_stft_mean'] = chroma_stft_mean  # chroma stft
    feature_set3s30s['chroma_stft_std'] = chroma_stft_std
    feature_set3s30s['chroma_stft_var'] = chroma_stft_var
    feature_set3s30s['chroma_cq_mean'] = chroma_cq_mean  # chroma cq
    feature_set3s30s['chroma_cq_std'] = chroma_cq_std
    feature_set3s30s['chroma_cq_var'] = chroma_cq_var
    feature_set3s30s['chroma_cens_mean'] = chroma_cens_mean  # chroma cens
    feature_set3s30s['chroma_cens_std'] = chroma_cens_std
    feature_set3s30s['chroma_cens_var'] = chroma_cens_var
    feature_set3s30s['melspectrogram_mean'] = mel_mean  # melspectrogram
    feature_set3s30s['melspectrogram_std'] = mel_std
    feature_set3s30s['melspectrogram_var'] = mel_var
#     feature_set3s30s['mfcc_mean'] = mfcc_mean  # mfcc
#     feature_set3s30s['mfcc_std'] = mfcc_std
#     feature_set3s30s['mfcc_var'] = mfcc_var
#     feature_set3s30s['mfcc_delta_mean'] = mfcc_delta_mean  # mfcc delta
#     feature_set3s30s['mfcc_delta_std'] = mfcc_delta_std
#     feature_set3s30s['mfcc_delta_var'] = mfcc_delta_var

    feature_set3s30s['mfcc1_mean'] = mfcc1_mean
    feature_set3s30s['mfcc2_mean'] = mfcc2_mean
    feature_set3s30s['mfcc3_mean'] = mfcc3_mean
    feature_set3s30s['mfcc4_mean'] = mfcc4_mean
    feature_set3s30s['mfcc5_mean'] = mfcc5_mean
    feature_set3s30s['mfcc6_mean'] = mfcc6_mean
    feature_set3s30s['mfcc7_mean'] = mfcc7_mean
    feature_set3s30s['mfcc8_mean'] = mfcc8_mean
    feature_set3s30s['mfcc9_mean'] = mfcc9_mean
    feature_set3s30s['mfcc10_mean'] = mfcc10_mean
    feature_set3s30s['mfcc11_mean'] = mfcc11_mean
    feature_set3s30s['mfcc12_mean'] = mfcc12_mean
    feature_set3s30s['mfcc13_mean'] = mfcc13_mean
    feature_set3s30s['mfcc14_mean'] = mfcc14_mean
    feature_set3s30s['mfcc15_mean'] = mfcc15_mean
    feature_set3s30s['mfcc16_mean'] = mfcc16_mean
    feature_set3s30s['mfcc17_mean'] = mfcc17_mean
    feature_set3s30s['mfcc18_mean'] = mfcc18_mean
    feature_set3s30s['mfcc19_mean'] = mfcc19_mean
    feature_set3s30s['mfcc20_mean'] = mfcc20_mean
    
    feature_set3s30s['mfcc1_std'] = mfcc1_std
    feature_set3s30s['mfcc2_std'] = mfcc2_std
    feature_set3s30s['mfcc3_std'] = mfcc3_std
    feature_set3s30s['mfcc4_std'] = mfcc4_std
    feature_set3s30s['mfcc5_std'] = mfcc5_std
    feature_set3s30s['mfcc6_std'] = mfcc6_std
    feature_set3s30s['mfcc7_std'] = mfcc7_std
    feature_set3s30s['mfcc8_std'] = mfcc8_std
    feature_set3s30s['mfcc9_std'] = mfcc9_std
    feature_set3s30s['mfcc10_std'] = mfcc10_std
    feature_set3s30s['mfcc11_std'] = mfcc11_std
    feature_set3s30s['mfcc12_std'] = mfcc12_std
    feature_set3s30s['mfcc13_std'] = mfcc13_std
    feature_set3s30s['mfcc14_std'] = mfcc14_std
    feature_set3s30s['mfcc15_std'] = mfcc15_std
    feature_set3s30s['mfcc16_std'] = mfcc16_std
    feature_set3s30s['mfcc17_std'] = mfcc17_std
    feature_set3s30s['mfcc18_std'] = mfcc18_std
    feature_set3s30s['mfcc19_std'] = mfcc19_std
    feature_set3s30s['mfcc20_std'] = mfcc20_std
    
    feature_set3s30s['mfcc1_var'] = mfcc1_var
    feature_set3s30s['mfcc2_var'] = mfcc2_var
    feature_set3s30s['mfcc3_var'] = mfcc3_var
    feature_set3s30s['mfcc4_var'] = mfcc4_var
    feature_set3s30s['mfcc5_var'] = mfcc5_var
    feature_set3s30s['mfcc6_var'] = mfcc6_var
    feature_set3s30s['mfcc7_var'] = mfcc7_var
    feature_set3s30s['mfcc8_var'] = mfcc8_var
    feature_set3s30s['mfcc9_var'] = mfcc9_var
    feature_set3s30s['mfcc10_var'] = mfcc10_var
    feature_set3s30s['mfcc11_var'] = mfcc11_var
    feature_set3s30s['mfcc12_var'] = mfcc12_var
    feature_set3s30s['mfcc13_var'] = mfcc13_var
    feature_set3s30s['mfcc14_var'] = mfcc14_var
    feature_set3s30s['mfcc15_var'] = mfcc15_var
    feature_set3s30s['mfcc16_var'] = mfcc16_var
    feature_set3s30s['mfcc17_var'] = mfcc17_var
    feature_set3s30s['mfcc18_var'] = mfcc18_var
    feature_set3s30s['mfcc19_var'] = mfcc19_var
    feature_set3s30s['mfcc20_var'] = mfcc20_var
    
    feature_set3s30s['mfcc1_delta_mean'] = mfcc1_delta_mean
    feature_set3s30s['mfcc2_delta_mean'] = mfcc2_delta_mean
    feature_set3s30s['mfcc3_delta_mean'] = mfcc3_delta_mean
    feature_set3s30s['mfcc4_delta_mean'] = mfcc4_delta_mean
    feature_set3s30s['mfcc5_delta_mean'] = mfcc5_delta_mean
    feature_set3s30s['mfcc6_delta_mean'] = mfcc6_delta_mean
    feature_set3s30s['mfcc7_delta_mean'] = mfcc7_delta_mean
    feature_set3s30s['mfcc8_delta_mean'] = mfcc8_delta_mean
    feature_set3s30s['mfcc9_delta_mean'] = mfcc9_delta_mean
    feature_set3s30s['mfcc10_delta_mean'] = mfcc10_delta_mean
    feature_set3s30s['mfcc11_delta_mean'] = mfcc11_delta_mean
    feature_set3s30s['mfcc12_delta_mean'] = mfcc12_delta_mean
    feature_set3s30s['mfcc13_delta_mean'] = mfcc13_delta_mean
    feature_set3s30s['mfcc14_delta_mean'] = mfcc14_delta_mean
    feature_set3s30s['mfcc15_delta_mean'] = mfcc15_delta_mean
    feature_set3s30s['mfcc16_delta_mean'] = mfcc16_delta_mean
    feature_set3s30s['mfcc17_delta_mean'] = mfcc17_delta_mean
    feature_set3s30s['mfcc18_delta_mean'] = mfcc18_delta_mean
    feature_set3s30s['mfcc19_delta_mean'] = mfcc19_delta_mean
    feature_set3s30s['mfcc20_delta_mean'] = mfcc20_delta_mean
    
    feature_set3s30s['mfcc1_delta_std'] = mfcc1_delta_std
    feature_set3s30s['mfcc2_delta_std'] = mfcc2_delta_std
    feature_set3s30s['mfcc3_delta_std'] = mfcc3_delta_std
    feature_set3s30s['mfcc4_delta_std'] = mfcc4_delta_std
    feature_set3s30s['mfcc5_delta_std'] = mfcc5_delta_std
    feature_set3s30s['mfcc6_delta_std'] = mfcc6_delta_std
    feature_set3s30s['mfcc7_delta_std'] = mfcc7_delta_std
    feature_set3s30s['mfcc8_delta_std'] = mfcc8_delta_std
    feature_set3s30s['mfcc9_delta_std'] = mfcc9_delta_std
    feature_set3s30s['mfcc10_delta_std'] = mfcc10_delta_std
    feature_set3s30s['mfcc11_delta_std'] = mfcc11_delta_std
    feature_set3s30s['mfcc12_delta_std'] = mfcc12_delta_std
    feature_set3s30s['mfcc13_delta_std'] = mfcc13_delta_std
    feature_set3s30s['mfcc14_delta_std'] = mfcc14_delta_std
    feature_set3s30s['mfcc15_delta_std'] = mfcc15_delta_std
    feature_set3s30s['mfcc16_delta_std'] = mfcc16_delta_std
    feature_set3s30s['mfcc17_delta_std'] = mfcc17_delta_std
    feature_set3s30s['mfcc18_delta_std'] = mfcc18_delta_std
    feature_set3s30s['mfcc19_delta_std'] = mfcc19_delta_std
    feature_set3s30s['mfcc20_delta_std'] = mfcc20_delta_std
    
    feature_set3s30s['mfcc1_delta_var'] = mfcc1_delta_var
    feature_set3s30s['mfcc2_delta_var'] = mfcc2_delta_var
    feature_set3s30s['mfcc3_delta_var'] = mfcc3_delta_var
    feature_set3s30s['mfcc4_delta_var'] = mfcc4_delta_var
    feature_set3s30s['mfcc5_delta_var'] = mfcc5_delta_var
    feature_set3s30s['mfcc6_delta_var'] = mfcc6_delta_var
    feature_set3s30s['mfcc7_delta_var'] = mfcc7_delta_var
    feature_set3s30s['mfcc8_delta_var'] = mfcc8_delta_var
    feature_set3s30s['mfcc9_delta_var'] = mfcc9_delta_var
    feature_set3s30s['mfcc10_delta_var'] = mfcc10_delta_var
    feature_set3s30s['mfcc11_delta_var'] = mfcc11_delta_var
    feature_set3s30s['mfcc12_delta_var'] = mfcc12_delta_var
    feature_set3s30s['mfcc13_delta_var'] = mfcc13_delta_var
    feature_set3s30s['mfcc14_delta_var'] = mfcc14_delta_var
    feature_set3s30s['mfcc15_delta_var'] = mfcc15_delta_var
    feature_set3s30s['mfcc16_delta_var'] = mfcc16_delta_var
    feature_set3s30s['mfcc17_delta_var'] = mfcc17_delta_var
    feature_set3s30s['mfcc18_delta_var'] = mfcc18_delta_var
    feature_set3s30s['mfcc19_delta_var'] = mfcc19_delta_var
    feature_set3s30s['mfcc20_delta_var'] = mfcc20_delta_var

    feature_set3s30s['rmse_mean'] = rmse_mean  # rmse
    feature_set3s30s['rmse_std'] = rmse_std
    feature_set3s30s['rmse_var'] = rmse_var
    feature_set3s30s['cent_mean'] = cent_mean  # cent
    feature_set3s30s['cent_std'] = cent_std
    feature_set3s30s['cent_var'] = cent_var
    feature_set3s30s['spec_bw_mean'] = spec_bw_mean  # spectral bandwidth
    feature_set3s30s['spec_bw_std'] = spec_bw_std
    feature_set3s30s['spec_bw_var'] = spec_bw_var
    feature_set3s30s['contrast_mean'] = contrast_mean  # contrast
    feature_set3s30s['contrast_std'] = contrast_std
    feature_set3s30s['contrast_var'] = contrast_var
    feature_set3s30s['rolloff_mean'] = rolloff_mean  # rolloff
    feature_set3s30s['rolloff_std'] = rolloff_std
    feature_set3s30s['rolloff_var'] = rolloff_var
    feature_set3s30s['poly_mean'] = poly_mean  # poly features
    feature_set3s30s['poly_std'] = poly_std
    feature_set3s30s['poly_var'] = poly_var
    feature_set3s30s['tonnetz_mean'] = tonnetz_mean  # tonnetz
    feature_set3s30s['tonnetz_std'] = tonnetz_std
    feature_set3s30s['tonnetz_var'] = tonnetz_var
    feature_set3s30s['zcr_mean'] = zcr_mean  # zero crossing rate
    feature_set3s30s['zcr_std'] = zcr_std
    feature_set3s30s['zcr_var'] = zcr_var
    feature_set3s30s['harm_mean'] = harm_mean  # harmonic
    feature_set3s30s['harm_std'] = harm_std
    feature_set3s30s['harm_var'] = harm_var
    feature_set3s30s['perc_mean'] = perc_mean  # percussive
    feature_set3s30s['perc_std'] = perc_std
    feature_set3s30s['perc_var'] = perc_var
    feature_set3s30s['frame_mean'] = frame_mean  # frames
    feature_set3s30s['frame_std'] = frame_std
    feature_set3s30s['frame_var'] = frame_var
    feature_set3s30s['songid']=feature_set3s30s['song_name'].str.split("-").str.get(0)
    feature_set3s30s['label'] = feature_set3s30s['song_name'].str.split("[0-9]").str.get(0)

    # 匯出成json及csv檔
    # feature_set3s30s.to_json('Emotion_features.json')
    dir30s = './csv30s'
    if not os.path.isdir(dir30s):
        os.mkdir(dir30s)

    feature_set3s30s.to_csv(f'{dir30s}/{file_name}30s.csv')
    
    # 特徵提取3s
    path = './musicfile/3s/%s/' %(file_name)
    id = 1  # Song ID
    
    # 給一個 DataFrame表格變數
    feature_set3s = pd.DataFrame()  
    
    # 給個別的特徵一個 Series欄位變數
    songname_vector = pd.Series(dtype='float64')
    tempo_vector = pd.Series(dtype='float64')
    total_beats = pd.Series(dtype='float64')
    average_beats = pd.Series(dtype='float64')
    chroma_stft_mean = pd.Series(dtype='float64')
    chroma_stft_std = pd.Series(dtype='float64')
    chroma_stft_var = pd.Series(dtype='float64')
    chroma_cq_mean = pd.Series(dtype='float64')
    chroma_cq_std = pd.Series(dtype='float64')
    chroma_cq_var = pd.Series(dtype='float64')
    chroma_cens_mean = pd.Series(dtype='float64')
    chroma_cens_std = pd.Series(dtype='float64')
    chroma_cens_var = pd.Series(dtype='float64')
    mel_mean = pd.Series(dtype='float64')
    mel_std = pd.Series(dtype='float64')
    mel_var = pd.Series(dtype='float64')
    
#     mfcc_mean = pd.Series()
#     mfcc_std = pd.Series()
#     mfcc_var = pd.Series()
#     mfcc_delta_mean = pd.Series()
#     mfcc_delta_std = pd.Series()
#     mfcc_delta_var = pd.Series()

    mfcc1_mean = pd.Series(dtype='float64')
    mfcc2_mean = pd.Series(dtype='float64')
    mfcc3_mean = pd.Series(dtype='float64')
    mfcc4_mean = pd.Series(dtype='float64')
    mfcc5_mean = pd.Series(dtype='float64')
    mfcc6_mean = pd.Series(dtype='float64')
    mfcc7_mean = pd.Series(dtype='float64')
    mfcc8_mean = pd.Series(dtype='float64')
    mfcc9_mean = pd.Series(dtype='float64')
    mfcc10_mean = pd.Series(dtype='float64')
    mfcc11_mean = pd.Series(dtype='float64')
    mfcc12_mean = pd.Series(dtype='float64')
    mfcc13_mean = pd.Series(dtype='float64')
    mfcc14_mean = pd.Series(dtype='float64')
    mfcc15_mean = pd.Series(dtype='float64')
    mfcc16_mean = pd.Series(dtype='float64')
    mfcc17_mean = pd.Series(dtype='float64')
    mfcc18_mean = pd.Series(dtype='float64')
    mfcc19_mean = pd.Series(dtype='float64')
    mfcc20_mean = pd.Series(dtype='float64')
    
    mfcc1_std = pd.Series(dtype='float64')
    mfcc2_std = pd.Series(dtype='float64')
    mfcc3_std = pd.Series(dtype='float64')
    mfcc4_std = pd.Series(dtype='float64')
    mfcc5_std = pd.Series(dtype='float64')
    mfcc6_std = pd.Series(dtype='float64')
    mfcc7_std = pd.Series(dtype='float64')
    mfcc8_std = pd.Series(dtype='float64')
    mfcc9_std = pd.Series(dtype='float64')
    mfcc10_std = pd.Series(dtype='float64')
    mfcc11_std = pd.Series(dtype='float64')
    mfcc12_std = pd.Series(dtype='float64')
    mfcc13_std = pd.Series(dtype='float64')
    mfcc14_std = pd.Series(dtype='float64')
    mfcc15_std = pd.Series(dtype='float64')
    mfcc16_std = pd.Series(dtype='float64')
    mfcc17_std = pd.Series(dtype='float64')
    mfcc18_std = pd.Series(dtype='float64')
    mfcc19_std = pd.Series(dtype='float64')
    mfcc20_std = pd.Series(dtype='float64')

    mfcc1_var = pd.Series(dtype='float64')
    mfcc2_var = pd.Series(dtype='float64')
    mfcc3_var = pd.Series(dtype='float64')
    mfcc4_var = pd.Series(dtype='float64')
    mfcc5_var = pd.Series(dtype='float64')
    mfcc6_var = pd.Series(dtype='float64')
    mfcc7_var = pd.Series(dtype='float64')
    mfcc8_var = pd.Series(dtype='float64')
    mfcc9_var = pd.Series(dtype='float64')
    mfcc10_var = pd.Series(dtype='float64')
    mfcc11_var = pd.Series(dtype='float64')
    mfcc12_var = pd.Series(dtype='float64')
    mfcc13_var = pd.Series(dtype='float64')
    mfcc14_var = pd.Series(dtype='float64')
    mfcc15_var = pd.Series(dtype='float64')
    mfcc16_var = pd.Series(dtype='float64')
    mfcc17_var = pd.Series(dtype='float64')
    mfcc18_var = pd.Series(dtype='float64')
    mfcc19_var = pd.Series(dtype='float64')
    mfcc20_var = pd.Series(dtype='float64')

    mfcc1_delta_mean = pd.Series(dtype='float64')
    mfcc2_delta_mean = pd.Series(dtype='float64')
    mfcc3_delta_mean = pd.Series(dtype='float64')
    mfcc4_delta_mean = pd.Series(dtype='float64')
    mfcc5_delta_mean = pd.Series(dtype='float64')
    mfcc6_delta_mean = pd.Series(dtype='float64')
    mfcc7_delta_mean = pd.Series(dtype='float64')
    mfcc8_delta_mean = pd.Series(dtype='float64')
    mfcc9_delta_mean = pd.Series(dtype='float64')
    mfcc10_delta_mean = pd.Series(dtype='float64')
    mfcc11_delta_mean = pd.Series(dtype='float64')
    mfcc12_delta_mean = pd.Series(dtype='float64')
    mfcc13_delta_mean = pd.Series(dtype='float64')
    mfcc14_delta_mean = pd.Series(dtype='float64')
    mfcc15_delta_mean = pd.Series(dtype='float64')
    mfcc16_delta_mean = pd.Series(dtype='float64')
    mfcc17_delta_mean = pd.Series(dtype='float64')
    mfcc18_delta_mean = pd.Series(dtype='float64')
    mfcc19_delta_mean = pd.Series(dtype='float64')
    mfcc20_delta_mean = pd.Series(dtype='float64')
    
    mfcc1_delta_std = pd.Series(dtype='float64')
    mfcc2_delta_std = pd.Series(dtype='float64')
    mfcc3_delta_std = pd.Series(dtype='float64')
    mfcc4_delta_std = pd.Series(dtype='float64')
    mfcc5_delta_std = pd.Series(dtype='float64')
    mfcc6_delta_std = pd.Series(dtype='float64')
    mfcc7_delta_std = pd.Series(dtype='float64')
    mfcc8_delta_std = pd.Series(dtype='float64')
    mfcc9_delta_std = pd.Series(dtype='float64')
    mfcc10_delta_std = pd.Series(dtype='float64')
    mfcc11_delta_std = pd.Series(dtype='float64')
    mfcc12_delta_std = pd.Series(dtype='float64')
    mfcc13_delta_std = pd.Series(dtype='float64')
    mfcc14_delta_std = pd.Series(dtype='float64')
    mfcc15_delta_std = pd.Series(dtype='float64')
    mfcc16_delta_std = pd.Series(dtype='float64')
    mfcc17_delta_std = pd.Series(dtype='float64')
    mfcc18_delta_std = pd.Series(dtype='float64')
    mfcc19_delta_std = pd.Series(dtype='float64')
    mfcc20_delta_std = pd.Series(dtype='float64')

    mfcc1_delta_var = pd.Series(dtype='float64')
    mfcc2_delta_var = pd.Series(dtype='float64')
    mfcc3_delta_var = pd.Series(dtype='float64')
    mfcc4_delta_var = pd.Series(dtype='float64')
    mfcc5_delta_var = pd.Series(dtype='float64')
    mfcc6_delta_var = pd.Series(dtype='float64')
    mfcc7_delta_var = pd.Series(dtype='float64')
    mfcc8_delta_var = pd.Series(dtype='float64')
    mfcc9_delta_var = pd.Series(dtype='float64')
    mfcc10_delta_var = pd.Series(dtype='float64')
    mfcc11_delta_var = pd.Series(dtype='float64')
    mfcc12_delta_var = pd.Series(dtype='float64')
    mfcc13_delta_var = pd.Series(dtype='float64')
    mfcc14_delta_var = pd.Series(dtype='float64')
    mfcc15_delta_var = pd.Series(dtype='float64')
    mfcc16_delta_var = pd.Series(dtype='float64')
    mfcc17_delta_var = pd.Series(dtype='float64')
    mfcc18_delta_var = pd.Series(dtype='float64')
    mfcc19_delta_var = pd.Series(dtype='float64')
    mfcc20_delta_var = pd.Series(dtype='float64')

    rmse_mean = pd.Series(dtype='float64')
    rmse_std = pd.Series(dtype='float64')
    rmse_var = pd.Series(dtype='float64')
    cent_mean = pd.Series(dtype='float64')
    cent_std = pd.Series(dtype='float64')
    cent_var = pd.Series(dtype='float64')
    spec_bw_mean = pd.Series(dtype='float64')
    spec_bw_std = pd.Series(dtype='float64')
    spec_bw_var = pd.Series(dtype='float64')
    contrast_mean = pd.Series(dtype='float64')
    contrast_std = pd.Series(dtype='float64')
    contrast_var = pd.Series(dtype='float64')
    rolloff_mean = pd.Series(dtype='float64')
    rolloff_std = pd.Series(dtype='float64')
    rolloff_var = pd.Series(dtype='float64')
    poly_mean = pd.Series(dtype='float64')
    poly_std = pd.Series(dtype='float64')
    poly_var = pd.Series(dtype='float64')
    tonnetz_mean = pd.Series(dtype='float64')
    tonnetz_std = pd.Series(dtype='float64')
    tonnetz_var = pd.Series(dtype='float64')
    zcr_mean = pd.Series(dtype='float64')
    zcr_std = pd.Series(dtype='float64')
    zcr_var = pd.Series(dtype='float64')
    harm_mean = pd.Series(dtype='float64')
    harm_std = pd.Series(dtype='float64')
    harm_var = pd.Series(dtype='float64')
    perc_mean = pd.Series(dtype='float64')
    perc_std = pd.Series(dtype='float64')
    perc_var = pd.Series(dtype='float64')
    frame_mean = pd.Series(dtype='float64')
    frame_std = pd.Series(dtype='float64')
    frame_var = pd.Series(dtype='float64')
    
    
    # 找出路徑資料夾下的所有檔案 ,輸出成 list
    file_data = [f for f in listdir(path) if isfile (join(path, f))]
    
    # list內的每個元素(資料長相:'檔名')取出來
    for line in file_data:
        if ( line[-1:] == '\n' ):   # 從換行前面取黨檔名
            line = line[:-1]     # 不取逗號

        # 讀取音檔
        songname = path + line # 將 目錄路徑跟檔名 合併成 檔案路徑
        y, sr = librosa.load(songname) # 用 librosa讀取檔案
        S = np.abs(librosa.stft(y)) # 傅立葉轉換取振幅
        
        # 特徵提取
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
        melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        rmse = librosa.feature.rms(y=y)
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        poly_features = librosa.feature.poly_features(S=S, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        harmonic = librosa.effects.harmonic(y)
        percussive = librosa.effects.percussive(y)
        
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        mfcc_delta = librosa.feature.delta(mfcc)
    
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        frames_to_time = librosa.frames_to_time(onset_frames[:20], sr=sr)
        
        # 將提取出各個特徵的值丟進前面建的 Series欄位變數
        songname_vector._set_value(id, line)  # song name
        tempo_vector._set_value(id, tempo)  # tempo
        total_beats._set_value(id, sum(beats))  # beats
        average_beats._set_value(id, np.average(beats))
        chroma_stft_mean._set_value(id, np.mean(chroma_stft))  # chroma stft
        chroma_stft_std._set_value(id, np.std(chroma_stft))
        chroma_stft_var._set_value(id, np.var(chroma_stft))
        chroma_cq_mean._set_value(id, np.mean(chroma_cq))  # chroma cq
        chroma_cq_std._set_value(id, np.std(chroma_cq))
        chroma_cq_var._set_value(id, np.var(chroma_cq))
        chroma_cens_mean._set_value(id, np.mean(chroma_cens))  # chroma cens
        chroma_cens_std._set_value(id, np.std(chroma_cens))
        chroma_cens_var._set_value(id, np.var(chroma_cens))
        mel_mean._set_value(id, np.mean(melspectrogram))  # melspectrogram
        mel_std._set_value(id, np.std(melspectrogram))
        mel_var._set_value(id, np.var(melspectrogram))
        
#         mfcc_mean._set_value(id, np.mean(mfcc))
#         mfcc_std._set_value(id, np.std(mfcc))
#         mfcc_var._set_value(id, np.var(mfcc))
#         mfcc_delta_mean._set_value(id, np.mean(mfcc_delta))  # mfcc delta
#         mfcc_delta_std._set_value(id, np.std(mfcc_delta))
#         mfcc_delta_var._set_value(id, np.var(mfcc_delta))

        mfcc1_mean._set_value(id, np.mean(mfcc[0]))
        mfcc2_mean._set_value(id, np.mean(mfcc[1]))
        mfcc3_mean._set_value(id, np.mean(mfcc[2]))
        mfcc4_mean._set_value(id, np.mean(mfcc[3]))
        mfcc5_mean._set_value(id, np.mean(mfcc[4]))
        mfcc6_mean._set_value(id, np.mean(mfcc[5]))
        mfcc7_mean._set_value(id, np.mean(mfcc[6]))
        mfcc8_mean._set_value(id, np.mean(mfcc[7]))
        mfcc9_mean._set_value(id, np.mean(mfcc[8]))
        mfcc10_mean._set_value(id, np.mean(mfcc[9]))
        mfcc11_mean._set_value(id, np.mean(mfcc[10]))
        mfcc12_mean._set_value(id, np.mean(mfcc[11]))
        mfcc13_mean._set_value(id, np.mean(mfcc[12]))
        mfcc14_mean._set_value(id, np.mean(mfcc[13]))
        mfcc15_mean._set_value(id, np.mean(mfcc[14]))
        mfcc16_mean._set_value(id, np.mean(mfcc[15]))
        mfcc17_mean._set_value(id, np.mean(mfcc[16]))
        mfcc18_mean._set_value(id, np.mean(mfcc[17]))
        mfcc19_mean._set_value(id, np.mean(mfcc[18]))
        mfcc20_mean._set_value(id, np.mean(mfcc[19]))
        
        mfcc1_std._set_value(id, np.std(mfcc[0]))
        mfcc2_std._set_value(id, np.std(mfcc[1]))
        mfcc3_std._set_value(id, np.std(mfcc[2]))
        mfcc4_std._set_value(id, np.std(mfcc[3]))
        mfcc5_std._set_value(id, np.std(mfcc[4]))
        mfcc6_std._set_value(id, np.std(mfcc[5]))
        mfcc7_std._set_value(id, np.std(mfcc[6]))
        mfcc8_std._set_value(id, np.std(mfcc[7]))
        mfcc9_std._set_value(id, np.std(mfcc[8]))
        mfcc10_std._set_value(id, np.std(mfcc[9]))
        mfcc11_std._set_value(id, np.std(mfcc[10]))
        mfcc12_std._set_value(id, np.std(mfcc[11]))
        mfcc13_std._set_value(id, np.std(mfcc[12]))
        mfcc14_std._set_value(id, np.std(mfcc[13]))
        mfcc15_std._set_value(id, np.std(mfcc[14]))
        mfcc16_std._set_value(id, np.std(mfcc[15]))
        mfcc17_std._set_value(id, np.std(mfcc[16]))
        mfcc18_std._set_value(id, np.std(mfcc[17]))
        mfcc19_std._set_value(id, np.std(mfcc[18]))
        mfcc20_std._set_value(id, np.std(mfcc[19]))
        
        mfcc1_var._set_value(id, np.var(mfcc[0]))
        mfcc2_var._set_value(id, np.var(mfcc[1]))
        mfcc3_var._set_value(id, np.var(mfcc[2]))
        mfcc4_var._set_value(id, np.var(mfcc[3]))
        mfcc5_var._set_value(id, np.var(mfcc[4]))
        mfcc6_var._set_value(id, np.var(mfcc[5]))
        mfcc7_var._set_value(id, np.var(mfcc[6]))
        mfcc8_var._set_value(id, np.var(mfcc[7]))
        mfcc9_var._set_value(id, np.var(mfcc[8]))
        mfcc10_var._set_value(id, np.var(mfcc[9]))
        mfcc11_var._set_value(id, np.var(mfcc[10]))
        mfcc12_var._set_value(id, np.var(mfcc[11]))
        mfcc13_var._set_value(id, np.var(mfcc[12]))
        mfcc14_var._set_value(id, np.var(mfcc[13]))
        mfcc15_var._set_value(id, np.var(mfcc[14]))
        mfcc16_var._set_value(id, np.var(mfcc[15]))
        mfcc17_var._set_value(id, np.var(mfcc[16]))
        mfcc18_var._set_value(id, np.var(mfcc[17]))
        mfcc19_var._set_value(id, np.var(mfcc[18]))
        mfcc20_var._set_value(id, np.var(mfcc[19]))
        
        mfcc1_delta_mean._set_value(id, np.mean(mfcc_delta[0]))
        mfcc2_delta_mean._set_value(id, np.mean(mfcc_delta[1]))
        mfcc3_delta_mean._set_value(id, np.mean(mfcc_delta[2]))
        mfcc4_delta_mean._set_value(id, np.mean(mfcc_delta[3]))
        mfcc5_delta_mean._set_value(id, np.mean(mfcc_delta[4]))
        mfcc6_delta_mean._set_value(id, np.mean(mfcc_delta[5]))
        mfcc7_delta_mean._set_value(id, np.mean(mfcc_delta[6]))
        mfcc8_delta_mean._set_value(id, np.mean(mfcc_delta[7]))
        mfcc9_delta_mean._set_value(id, np.mean(mfcc_delta[8]))
        mfcc10_delta_mean._set_value(id, np.mean(mfcc_delta[9]))
        mfcc11_delta_mean._set_value(id, np.mean(mfcc_delta[10]))
        mfcc12_delta_mean._set_value(id, np.mean(mfcc_delta[11]))
        mfcc13_delta_mean._set_value(id, np.mean(mfcc_delta[12]))
        mfcc14_delta_mean._set_value(id, np.mean(mfcc_delta[13]))
        mfcc15_delta_mean._set_value(id, np.mean(mfcc_delta[14]))
        mfcc16_delta_mean._set_value(id, np.mean(mfcc_delta[15]))
        mfcc17_delta_mean._set_value(id, np.mean(mfcc_delta[16]))
        mfcc18_delta_mean._set_value(id, np.mean(mfcc_delta[17]))
        mfcc19_delta_mean._set_value(id, np.mean(mfcc_delta[18]))
        mfcc20_delta_mean._set_value(id, np.mean(mfcc_delta[19]))
        
        mfcc1_delta_std._set_value(id, np.std(mfcc_delta[0]))
        mfcc2_delta_std._set_value(id, np.std(mfcc_delta[1]))
        mfcc3_delta_std._set_value(id, np.std(mfcc_delta[2]))
        mfcc4_delta_std._set_value(id, np.std(mfcc_delta[3]))
        mfcc5_delta_std._set_value(id, np.std(mfcc_delta[4]))
        mfcc6_delta_std._set_value(id, np.std(mfcc_delta[5]))
        mfcc7_delta_std._set_value(id, np.std(mfcc_delta[6]))
        mfcc8_delta_std._set_value(id, np.std(mfcc_delta[7]))
        mfcc9_delta_std._set_value(id, np.std(mfcc_delta[8]))
        mfcc10_delta_std._set_value(id, np.std(mfcc_delta[9]))
        mfcc11_delta_std._set_value(id, np.std(mfcc_delta[10]))
        mfcc12_delta_std._set_value(id, np.std(mfcc_delta[11]))
        mfcc13_delta_std._set_value(id, np.std(mfcc_delta[12]))
        mfcc14_delta_std._set_value(id, np.std(mfcc_delta[13]))
        mfcc15_delta_std._set_value(id, np.std(mfcc_delta[14]))
        mfcc16_delta_std._set_value(id, np.std(mfcc_delta[15]))
        mfcc17_delta_std._set_value(id, np.std(mfcc_delta[16]))
        mfcc18_delta_std._set_value(id, np.std(mfcc_delta[17]))
        mfcc19_delta_std._set_value(id, np.std(mfcc_delta[18]))
        mfcc20_delta_std._set_value(id, np.std(mfcc_delta[19]))
        
        mfcc1_delta_var._set_value(id, np.var(mfcc_delta[0]))
        mfcc2_delta_var._set_value(id, np.var(mfcc_delta[1]))
        mfcc3_delta_var._set_value(id, np.var(mfcc_delta[2]))
        mfcc4_delta_var._set_value(id, np.var(mfcc_delta[3]))
        mfcc5_delta_var._set_value(id, np.var(mfcc_delta[4]))
        mfcc6_delta_var._set_value(id, np.var(mfcc_delta[5]))
        mfcc7_delta_var._set_value(id, np.var(mfcc_delta[6]))
        mfcc8_delta_var._set_value(id, np.var(mfcc_delta[7]))
        mfcc9_delta_var._set_value(id, np.var(mfcc_delta[8]))
        mfcc10_delta_var._set_value(id, np.var(mfcc_delta[9]))
        mfcc11_delta_var._set_value(id, np.var(mfcc_delta[10]))
        mfcc12_delta_var._set_value(id, np.var(mfcc_delta[11]))
        mfcc13_delta_var._set_value(id, np.var(mfcc_delta[12]))
        mfcc14_delta_var._set_value(id, np.var(mfcc_delta[13]))
        mfcc15_delta_var._set_value(id, np.var(mfcc_delta[14]))
        mfcc16_delta_var._set_value(id, np.var(mfcc_delta[15]))
        mfcc17_delta_var._set_value(id, np.var(mfcc_delta[16]))
        mfcc18_delta_var._set_value(id, np.var(mfcc_delta[17]))
        mfcc19_delta_var._set_value(id, np.var(mfcc_delta[18]))
        mfcc20_delta_var._set_value(id, np.var(mfcc_delta[19]))
        
        rmse_mean._set_value(id, np.mean(rmse))  # rmse
        rmse_std._set_value(id, np.std(rmse))
        rmse_var._set_value(id, np.var(rmse))
        cent_mean._set_value(id, np.mean(cent))  # cent
        cent_std._set_value(id, np.std(cent))
        cent_var._set_value(id, np.var(cent))
        spec_bw_mean._set_value(id, np.mean(spec_bw))  # spectral bandwidth
        spec_bw_std._set_value(id, np.std(spec_bw))
        spec_bw_var._set_value(id, np.var(spec_bw))
        contrast_mean._set_value(id, np.mean(contrast))  # contrast
        contrast_std._set_value(id, np.std(contrast))
        contrast_var._set_value(id, np.var(contrast))
        rolloff_mean._set_value(id, np.mean(rolloff))  # rolloff
        rolloff_std._set_value(id, np.std(rolloff))
        rolloff_var._set_value(id, np.var(rolloff))
        poly_mean._set_value(id, np.mean(poly_features))  # poly features
        poly_std._set_value(id, np.std(poly_features))
        poly_var._set_value(id, np.var(poly_features))
        tonnetz_mean._set_value(id, np.mean(tonnetz))  # tonnetz
        tonnetz_std._set_value(id, np.std(tonnetz))
        tonnetz_var._set_value(id, np.var(tonnetz))
        zcr_mean._set_value(id, np.mean(zcr))  # zero crossing rate
        zcr_std._set_value(id, np.std(zcr))
        zcr_var._set_value(id, np.var(zcr))
        harm_mean._set_value(id, np.mean(harmonic))  # harmonic
        harm_std._set_value(id, np.std(harmonic))
        harm_var._set_value(id, np.var(harmonic))
        perc_mean._set_value(id, np.mean(percussive))  # percussive
        perc_std._set_value(id, np.std(percussive))
        perc_var._set_value(id, np.var(percussive))
        frame_mean._set_value(id, np.mean(frames_to_time))  # frames
        frame_std._set_value(id, np.std(frames_to_time))
        frame_var._set_value(id, np.var(frames_to_time))
        
        
        id = id+1   #完成換下一首歌
    
    # 將各個 Series 加入前面建的 feature_set3s dataframe裡
    feature_set3s['song_name'] = songname_vector  # song name
    feature_set3s['tempo'] = tempo_vector  # tempo 
    feature_set3s['total_beats'] = total_beats  # beats
    feature_set3s['average_beats'] = average_beats
    feature_set3s['chroma_stft_mean'] = chroma_stft_mean  # chroma stft
    feature_set3s['chroma_stft_std'] = chroma_stft_std
    feature_set3s['chroma_stft_var'] = chroma_stft_var
    feature_set3s['chroma_cq_mean'] = chroma_cq_mean  # chroma cq
    feature_set3s['chroma_cq_std'] = chroma_cq_std
    feature_set3s['chroma_cq_var'] = chroma_cq_var
    feature_set3s['chroma_cens_mean'] = chroma_cens_mean  # chroma cens
    feature_set3s['chroma_cens_std'] = chroma_cens_std
    feature_set3s['chroma_cens_var'] = chroma_cens_var
    feature_set3s['melspectrogram_mean'] = mel_mean  # melspectrogram
    feature_set3s['melspectrogram_std'] = mel_std
    feature_set3s['melspectrogram_var'] = mel_var
#     feature_set3s['mfcc_mean'] = mfcc_mean  # mfcc
#     feature_set3s['mfcc_std'] = mfcc_std
#     feature_set3s['mfcc_var'] = mfcc_var
#     feature_set3s['mfcc_delta_mean'] = mfcc_delta_mean  # mfcc delta
#     feature_set3s['mfcc_delta_std'] = mfcc_delta_std
#     feature_set3s['mfcc_delta_var'] = mfcc_delta_var

    feature_set3s['mfcc1_mean'] = mfcc1_mean
    feature_set3s['mfcc2_mean'] = mfcc2_mean
    feature_set3s['mfcc3_mean'] = mfcc3_mean
    feature_set3s['mfcc4_mean'] = mfcc4_mean
    feature_set3s['mfcc5_mean'] = mfcc5_mean
    feature_set3s['mfcc6_mean'] = mfcc6_mean
    feature_set3s['mfcc7_mean'] = mfcc7_mean
    feature_set3s['mfcc8_mean'] = mfcc8_mean
    feature_set3s['mfcc9_mean'] = mfcc9_mean
    feature_set3s['mfcc10_mean'] = mfcc10_mean
    feature_set3s['mfcc11_mean'] = mfcc11_mean
    feature_set3s['mfcc12_mean'] = mfcc12_mean
    feature_set3s['mfcc13_mean'] = mfcc13_mean
    feature_set3s['mfcc14_mean'] = mfcc14_mean
    feature_set3s['mfcc15_mean'] = mfcc15_mean
    feature_set3s['mfcc16_mean'] = mfcc16_mean
    feature_set3s['mfcc17_mean'] = mfcc17_mean
    feature_set3s['mfcc18_mean'] = mfcc18_mean
    feature_set3s['mfcc19_mean'] = mfcc19_mean
    feature_set3s['mfcc20_mean'] = mfcc20_mean
    
    feature_set3s['mfcc1_std'] = mfcc1_std
    feature_set3s['mfcc2_std'] = mfcc2_std
    feature_set3s['mfcc3_std'] = mfcc3_std
    feature_set3s['mfcc4_std'] = mfcc4_std
    feature_set3s['mfcc5_std'] = mfcc5_std
    feature_set3s['mfcc6_std'] = mfcc6_std
    feature_set3s['mfcc7_std'] = mfcc7_std
    feature_set3s['mfcc8_std'] = mfcc8_std
    feature_set3s['mfcc9_std'] = mfcc9_std
    feature_set3s['mfcc10_std'] = mfcc10_std
    feature_set3s['mfcc11_std'] = mfcc11_std
    feature_set3s['mfcc12_std'] = mfcc12_std
    feature_set3s['mfcc13_std'] = mfcc13_std
    feature_set3s['mfcc14_std'] = mfcc14_std
    feature_set3s['mfcc15_std'] = mfcc15_std
    feature_set3s['mfcc16_std'] = mfcc16_std
    feature_set3s['mfcc17_std'] = mfcc17_std
    feature_set3s['mfcc18_std'] = mfcc18_std
    feature_set3s['mfcc19_std'] = mfcc19_std
    feature_set3s['mfcc20_std'] = mfcc20_std
    
    feature_set3s['mfcc1_var'] = mfcc1_var
    feature_set3s['mfcc2_var'] = mfcc2_var
    feature_set3s['mfcc3_var'] = mfcc3_var
    feature_set3s['mfcc4_var'] = mfcc4_var
    feature_set3s['mfcc5_var'] = mfcc5_var
    feature_set3s['mfcc6_var'] = mfcc6_var
    feature_set3s['mfcc7_var'] = mfcc7_var
    feature_set3s['mfcc8_var'] = mfcc8_var
    feature_set3s['mfcc9_var'] = mfcc9_var
    feature_set3s['mfcc10_var'] = mfcc10_var
    feature_set3s['mfcc11_var'] = mfcc11_var
    feature_set3s['mfcc12_var'] = mfcc12_var
    feature_set3s['mfcc13_var'] = mfcc13_var
    feature_set3s['mfcc14_var'] = mfcc14_var
    feature_set3s['mfcc15_var'] = mfcc15_var
    feature_set3s['mfcc16_var'] = mfcc16_var
    feature_set3s['mfcc17_var'] = mfcc17_var
    feature_set3s['mfcc18_var'] = mfcc18_var
    feature_set3s['mfcc19_var'] = mfcc19_var
    feature_set3s['mfcc20_var'] = mfcc20_var
    
    feature_set3s['mfcc1_delta_mean'] = mfcc1_delta_mean
    feature_set3s['mfcc2_delta_mean'] = mfcc2_delta_mean
    feature_set3s['mfcc3_delta_mean'] = mfcc3_delta_mean
    feature_set3s['mfcc4_delta_mean'] = mfcc4_delta_mean
    feature_set3s['mfcc5_delta_mean'] = mfcc5_delta_mean
    feature_set3s['mfcc6_delta_mean'] = mfcc6_delta_mean
    feature_set3s['mfcc7_delta_mean'] = mfcc7_delta_mean
    feature_set3s['mfcc8_delta_mean'] = mfcc8_delta_mean
    feature_set3s['mfcc9_delta_mean'] = mfcc9_delta_mean
    feature_set3s['mfcc10_delta_mean'] = mfcc10_delta_mean
    feature_set3s['mfcc11_delta_mean'] = mfcc11_delta_mean
    feature_set3s['mfcc12_delta_mean'] = mfcc12_delta_mean
    feature_set3s['mfcc13_delta_mean'] = mfcc13_delta_mean
    feature_set3s['mfcc14_delta_mean'] = mfcc14_delta_mean
    feature_set3s['mfcc15_delta_mean'] = mfcc15_delta_mean
    feature_set3s['mfcc16_delta_mean'] = mfcc16_delta_mean
    feature_set3s['mfcc17_delta_mean'] = mfcc17_delta_mean
    feature_set3s['mfcc18_delta_mean'] = mfcc18_delta_mean
    feature_set3s['mfcc19_delta_mean'] = mfcc19_delta_mean
    feature_set3s['mfcc20_delta_mean'] = mfcc20_delta_mean
    
    feature_set3s['mfcc1_delta_std'] = mfcc1_delta_std
    feature_set3s['mfcc2_delta_std'] = mfcc2_delta_std
    feature_set3s['mfcc3_delta_std'] = mfcc3_delta_std
    feature_set3s['mfcc4_delta_std'] = mfcc4_delta_std
    feature_set3s['mfcc5_delta_std'] = mfcc5_delta_std
    feature_set3s['mfcc6_delta_std'] = mfcc6_delta_std
    feature_set3s['mfcc7_delta_std'] = mfcc7_delta_std
    feature_set3s['mfcc8_delta_std'] = mfcc8_delta_std
    feature_set3s['mfcc9_delta_std'] = mfcc9_delta_std
    feature_set3s['mfcc10_delta_std'] = mfcc10_delta_std
    feature_set3s['mfcc11_delta_std'] = mfcc11_delta_std
    feature_set3s['mfcc12_delta_std'] = mfcc12_delta_std
    feature_set3s['mfcc13_delta_std'] = mfcc13_delta_std
    feature_set3s['mfcc14_delta_std'] = mfcc14_delta_std
    feature_set3s['mfcc15_delta_std'] = mfcc15_delta_std
    feature_set3s['mfcc16_delta_std'] = mfcc16_delta_std
    feature_set3s['mfcc17_delta_std'] = mfcc17_delta_std
    feature_set3s['mfcc18_delta_std'] = mfcc18_delta_std
    feature_set3s['mfcc19_delta_std'] = mfcc19_delta_std
    feature_set3s['mfcc20_delta_std'] = mfcc20_delta_std
    
    feature_set3s['mfcc1_delta_var'] = mfcc1_delta_var
    feature_set3s['mfcc2_delta_var'] = mfcc2_delta_var
    feature_set3s['mfcc3_delta_var'] = mfcc3_delta_var
    feature_set3s['mfcc4_delta_var'] = mfcc4_delta_var
    feature_set3s['mfcc5_delta_var'] = mfcc5_delta_var
    feature_set3s['mfcc6_delta_var'] = mfcc6_delta_var
    feature_set3s['mfcc7_delta_var'] = mfcc7_delta_var
    feature_set3s['mfcc8_delta_var'] = mfcc8_delta_var
    feature_set3s['mfcc9_delta_var'] = mfcc9_delta_var
    feature_set3s['mfcc10_delta_var'] = mfcc10_delta_var
    feature_set3s['mfcc11_delta_var'] = mfcc11_delta_var
    feature_set3s['mfcc12_delta_var'] = mfcc12_delta_var
    feature_set3s['mfcc13_delta_var'] = mfcc13_delta_var
    feature_set3s['mfcc14_delta_var'] = mfcc14_delta_var
    feature_set3s['mfcc15_delta_var'] = mfcc15_delta_var
    feature_set3s['mfcc16_delta_var'] = mfcc16_delta_var
    feature_set3s['mfcc17_delta_var'] = mfcc17_delta_var
    feature_set3s['mfcc18_delta_var'] = mfcc18_delta_var
    feature_set3s['mfcc19_delta_var'] = mfcc19_delta_var
    feature_set3s['mfcc20_delta_var'] = mfcc20_delta_var

    feature_set3s['rmse_mean'] = rmse_mean  # rmse
    feature_set3s['rmse_std'] = rmse_std
    feature_set3s['rmse_var'] = rmse_var
    feature_set3s['cent_mean'] = cent_mean  # cent
    feature_set3s['cent_std'] = cent_std
    feature_set3s['cent_var'] = cent_var
    feature_set3s['spec_bw_mean'] = spec_bw_mean  # spectral bandwidth
    feature_set3s['spec_bw_std'] = spec_bw_std
    feature_set3s['spec_bw_var'] = spec_bw_var
    feature_set3s['contrast_mean'] = contrast_mean  # contrast
    feature_set3s['contrast_std'] = contrast_std
    feature_set3s['contrast_var'] = contrast_var
    feature_set3s['rolloff_mean'] = rolloff_mean  # rolloff
    feature_set3s['rolloff_std'] = rolloff_std
    feature_set3s['rolloff_var'] = rolloff_var
    feature_set3s['poly_mean'] = poly_mean  # poly features
    feature_set3s['poly_std'] = poly_std
    feature_set3s['poly_var'] = poly_var
    feature_set3s['tonnetz_mean'] = tonnetz_mean  # tonnetz
    feature_set3s['tonnetz_std'] = tonnetz_std
    feature_set3s['tonnetz_var'] = tonnetz_var
    feature_set3s['zcr_mean'] = zcr_mean  # zero crossing rate
    feature_set3s['zcr_std'] = zcr_std
    feature_set3s['zcr_var'] = zcr_var
    feature_set3s['harm_mean'] = harm_mean  # harmonic
    feature_set3s['harm_std'] = harm_std
    feature_set3s['harm_var'] = harm_var
    feature_set3s['perc_mean'] = perc_mean  # percussive
    feature_set3s['perc_std'] = perc_std
    feature_set3s['perc_var'] = perc_var
    feature_set3s['frame_mean'] = frame_mean  # frames
    feature_set3s['frame_std'] = frame_std
    feature_set3s['frame_var'] = frame_var
    feature_set3s['songid']=feature_set3s['song_name'].str.split("-").str.get(0)
    feature_set3s['label'] = feature_set3s['song_name'].str.split("[0-9]").str.get(0)

    # 匯出成json及csv檔
    # feature_set3s.to_json('Emotion_features.json')
    dir3s = './csv3s'
    if not os.path.isdir(dir3s):
        os.mkdir(dir3s)
    
    feature_set3s.to_csv(f'{dir3s}/{file_name}3s.csv')
                
file_name = sys.argv[1]
preprocessing(file_name)                    
                    
