import librosa
import pandas as pd
import numpy as np
from IPython.display import Audio  #播放套件
import matplotlib.pyplot as plt 
import librosa.display  #libroso 繪圖
import os
from os import listdir
from os.path import isfile, join
import sys

# 路徑
# songname = 'regaeeee'
def print_ft_jpg(songname):
    # 讀取音樂
    path = f'./musicfile/wav/{songname}/{songname}.wav'
    # 特徵圖匯出路徑
    parentpath = './ft_jpg/'
    ft_jpg_output = f'./ft_jpg/{songname}/'
    if not os.path.isdir(parentpath):
        os.mkdir(parentpath)
    if not os.path.isdir(ft_jpg_output):
        os.mkdir(ft_jpg_output)

    # 提取user music librosa y
    y, sr = librosa.load(path, duration=120)
    S = np.abs(librosa.stft(y))


    # print each feature pic
    ### 繪製波形圖
    fig, ax = plt.subplots(sharex=True, figsize=(12,1))  # matplotlib 開圖面
    librosa.display.waveshow(y, sr=sr)  # 波形繪製

    plt.savefig(f'{ft_jpg_output}{songname}_wav_plot.jpg')
    print("wav_plot complete.")
    ### 頻譜圖  Ex: Short-time Fourier transform (STFT) 短時傅立葉轉換)
    STFT = librosa.stft(y=y)
    fig, ax = plt.subplots(figsize=(10,5))
    STFT_abs = np.abs(STFT)   # STFT 取絕對值，振幅的絕對值

    # librosa.display 繪製光譜圖，搭配 matplotlib 套件繪製。
    img = librosa.display.specshow(librosa.amplitude_to_db(STFT_abs,ref=np.max), # amplitude_to_db() 振幅轉分貝
                                y_axis='log', x_axis='time', ax=ax)
    ax.set_title('STFT Power spectrogram') # 標題
    fig.colorbar(img, ax=ax, format="%+2.0f dB") # 分貝標示

    plt.savefig(f'{ft_jpg_output}{songname}_STFT.jpg')
    print("STFT complete.")
    # 特徵提取

    # 偵測節拍，輸出兩個變數
    # ----- y:音訊資料點, sr:採樣率, hop_length(不寫預設 512)-----
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=512) 
    hop_length = 512 # 預設 512，跳躍長度? 越大頻譜圖越壓縮
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median) # 取光譜強度值
    times = librosa.times_like(onset_env, sr=sr, hop_length=hop_length) # 取出時間資訊

    # 開兩張圖對比
    fig, ax = plt.subplots(nrows=2, sharex=True ,figsize=(24,6)) # 開兩張畫布，nrow為畫布數量，ax為第幾張子圖
    ax[0].plot(times, librosa.util.normalize(onset_env),label='Onset strength') # 將 onset_env value 正歸化成 0~1 範圍
    ax[0].vlines(times[beat_frames], 0, 1, alpha=0.5, color='r',linestyle='--', label='Beats: '+str(len(beat_frames)))
    librosa.display.waveshow(y, sr=sr,ax=ax[1], label='wave') # 第二張波形圖
    ax[0].legend() # 第一章子圖加上標籤
    ax[1].legend() # 第二張子圖加上標籤
    plt.savefig(f'{ft_jpg_output}{songname}_onset.jpg')
    print("onset complete.")

    S = np.abs(librosa.stft(y)) # 先轉為 stft
    chroma_stft = librosa.feature.chroma_stft(S=S, sr=sr) 
    chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
    fig, ax = plt.subplots(nrows=3,ncols=1, sharex=True, sharey=True, figsize=(15, 10))
    librosa.display.specshow(chroma_stft, y_axis='chroma', x_axis='time', ax=ax[0])
    librosa.display.specshow(chroma_cq, y_axis='chroma', x_axis='time', ax=ax[1])
    librosa.display.specshow(chroma_cens, y_axis='chroma', x_axis='time', ax=ax[2])
    ax[0].set(title='chroma_stft')
    ax[1].set(title='chroma_cq')
    ax[2].set(title='chroma_cens')
    fig.colorbar(img, ax=ax)
    plt.savefig(f'{ft_jpg_output}{songname}_3chroma.jpg')
    print("3chroma complete.")

    ## Ｍel-spectrogram 梅爾尺度光譜 ＆ MFCC 梅爾倒頻係數

    melspectrogram = librosa.feature.melspectrogram(S=S, sr=sr)

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20) # n_mfcc 數量
    fig, ax = plt.subplots(nrows=2,ncols=1, sharex=True, figsize=(10, 7))
    img_mel = librosa.display.specshow(librosa.power_to_db(melspectrogram, ref=np.max),
                                x_axis='time', y_axis='mel', fmax=10000, ax=ax[0])
    img_mfcc= librosa.display.specshow(mfccs, x_axis='time',y_axis='mel', ax=ax[1])
    ax[0].label_outer()
    ax[0].set(title='Mel spectrogram')
    ax[1].set(title='MFCC')
    fig.colorbar(img_mel, ax=[ax[0]])
    fig.colorbar(img_mfcc, ax=[ax[1]])
    plt.savefig(f'{ft_jpg_output}{songname}_mfcc.jpg')
    print("mfcc complete.")

    rms = librosa.feature.rms(y=y)
    times = librosa.times_like(rms)  # 提取時間
    S, phase = librosa.magphase(librosa.stft(y))  # 只取 S,  (D = S*P )

    fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(10,5))
    ax[0].semilogy(times, rms[0], label='RMS Energy')   # rms => (1, 130), rms[0] => (130,)
    ax[0].legend()
    ax[0].label_outer()
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                            y_axis='log', x_axis='time', ax=ax[1])
    ax[1].set(title='STFT log Power spectrogram')
    plt.savefig(f'{ft_jpg_output}{songname}_rmse.jpg')
    print("rmse complete.")

    # 方法1. 使用時間序列計算
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    # 方法2. 使用頻譜計算
    D = librosa.stft(y)  # stft
    S, phase = librosa.magphase(D)  # 只取 S,  (D = S*P )
    cent1 = librosa.feature.spectral_centroid(S=S)
    times = librosa.times_like(cent) # 提取時間
    fig, ax = plt.subplots(figsize=(10,4)) # 開圖面設定大小
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(D), ref=np.max),
                            y_axis='log', x_axis='time', ax=ax)
    ax.plot(times, cent[0], label='Spectral centroid', color='w')
    ax.legend(loc='upper right')
    ax.set(title='log Power spectrogram')

    plt.savefig(f'{ft_jpg_output}{songname}_spec_cen.jpg')
    print("spec_cen complete.")

    # 2.頻譜圖輸入
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    S, phase = librosa.magphase(librosa.stft(y=y))
    times = librosa.times_like(spec_bw)
    centroid = librosa.feature.spectral_centroid(S=S)

    fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(10,8))
    # ax[0].semilogy(times, spec_bw[0], label='Spectral bandwidth')
    ax[0].plot(times, spec_bw[0],label='Spectral bandwidth')
    ax[0].set(ylabel='Hz', xticks=[], xlim=[times.min(), times.max()])
    ax[0].legend()
    ax[0].label_outer()

    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                            y_axis='log', x_axis='time', ax=ax[1])

    ax[1].fill_between(times,
                    np.maximum(0, centroid[0] - spec_bw[0]),
                    np.minimum(centroid[0] + spec_bw[0], sr/2),
                    alpha=0.8, label='Centroid +- bandwidth')
    ax[1].plot(times, spec_bw[0], label='Spectral centroid', color='w')
    ax[1].legend(loc='lower right')

    plt.savefig(f'{ft_jpg_output}{songname}_spec_bd.jpg')
    print("spec_bd complete.")

    # 輸入 stft取絕對值
    S = np.abs(librosa.stft(y))
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
    fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(11,8))
    img1 = librosa.display.specshow(chroma_stft, y_axis='chroma', x_axis='time', ax=ax[0])

    fig.colorbar(img1, ax=[ax[0]], format='%+2.0f dB')
    ax[0].set(title='chroma_stft')
    ax[0].label_outer()

    img2 = librosa.display.specshow(contrast, x_axis='time', ax=ax[1])
    fig.colorbar(img2, ax=[ax[1]])
    ax[1].set(ylabel='Frequency bands', title='Spectral contrast')

    plt.savefig(f'{ft_jpg_output}{songname}_spec_contrast.jpg')
    print("spec_constrast complete.")

    # y = librosa.effects.harmonic(y)  # 從時間序列中提取諧波元素
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(11,8))
    img1 = librosa.display.specshow(tonnetz,
                                    y_axis='tonnetz', x_axis='time', ax=ax[0])
    img2 = librosa.display.specshow(librosa.feature.chroma_cqt(y=y, sr=sr),
                                    y_axis='chroma', x_axis='time', ax=ax[1])
    ax[0].set(title='Tonal Centroids (Tonnetz)')
    ax[0].label_outer()
    ax[1].set(title='Chroma')
    fig.colorbar(img1, ax=[ax[0]])
    fig.colorbar(img2, ax=[ax[1]])

    plt.savefig(f'{ft_jpg_output}{songname}_tonnetz.jpg')
    print("tonnetz complete.")

    # zcr 過零率
    zcr = librosa.feature.zero_crossing_rate(y=y)
    librosa.display.waveshow(zcr, sr=sr, alpha=1, label='zcr')

    plt.savefig(f'{ft_jpg_output}{songname}_zcr.jpg')
    print("zcr complete.")

    ## effects.harmonic 諧波特徵

    # Use a margin > 1.0 for greater harmonic separation 使用更大的諧波分離
    y_harmonic_m = librosa.effects.harmonic(y, margin=5.0)
    # 諧波變化波形
    fig, ax = plt.subplots(figsize=(50,4))
    librosa.display.waveshow(y, sr=sr, ax=ax, label='Original wave')
    librosa.display.waveshow(y_harmonic_m, sr=sr, ax=ax, alpha=1, label='harmonic (margin: 5)')
    ax.legend() # 加上標籤

    plt.savefig(f'{ft_jpg_output}{songname}_harmonic.jpg')
    print("harmonic complete.")

    # 諧波頻譜變化
    ys_harmonic_m = librosa.stft(y_harmonic_m)
    fig, ax = plt.subplots(nrows = 2, sharex=True, figsize=(11,10))
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                            y_axis='log', x_axis='time', ax=ax[0])
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(ys_harmonic_m), ref=np.max),
                            y_axis='log', x_axis='time', ax=ax[1])
    ax[0].set(title='Original - log Power spectrogram')
    ax[1].set(title='harmonic (margin=5.0)')
    ax[0].label_outer()
    ax[1].label_outer()

    plt.savefig(f'{ft_jpg_output}{songname}_harmonic_spec.jpg')
    print("harnonic_spec complete.")

    ## effects.percussive 打擊特徵 （鼓點）

    y_percussive_m = librosa.effects.percussive(y, margin=5.0) #使用更大的效果

    # 打擊特徵分離出的波形
    fig, ax = plt.subplots(figsize=(20,4))
    librosa.display.waveshow(y_harmonic_m, sr=sr, ax=ax, label='harmonic (default)', color='red')
    librosa.display.waveshow(y_percussive_m, sr=sr, ax=ax, label='precussive (margin=5.0)')
    ax.legend(loc='upper left')


    plt.savefig(f'{ft_jpg_output}{songname}_percussive.jpg')
    print("percussive complete.")

    # 打擊特徵分離出的頻譜圖
    ys_percussive_m = librosa.stft(y_percussive_m)
    fig, ax = plt.subplots(nrows = 2, figsize=(11,10))
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                            y_axis='log', x_axis='time', ax=ax[0])
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(ys_percussive_m), ref=np.max),
                            y_axis='log', x_axis='time', ax=ax[1])
    ax[0].set(title='Original - log Power spectrogram')
    # ax[1].set(title='percussive (default)')
    ax[1].set(title='percussive (margin=5.0)')
    ax[0].label_outer()
    ax[1].label_outer()

    plt.savefig(f'{ft_jpg_output}{songname}_percussive_spec.jpg')
    print("percussive_spec complete.")

    # 近似最大頻率滾降百分比 =0.85 (default)
    rolloff_def = librosa.feature.spectral_rolloff(y=y, sr=sr)
    # 近似最大頻率滾降百分比 =0.99
    rolloff_max = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.50)
    # 近似最小頻率滾降百分比 =0.01
    rolloff_min = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.01)
    S, phase = librosa.magphase(librosa.stft(y)) # stft處理 -> 拆分振幅和相位
    fig, ax = plt.subplots(nrows = 2, figsize=(11,6))
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), y_axis='log', x_axis='time', ax=ax[0])
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), y_axis='log', x_axis='time', ax=ax[1])
    ax[1].plot(librosa.times_like(rolloff_def), rolloff_def[0], label='Roll-off frequency (0.85)')
    ax[1].plot(librosa.times_like(rolloff_max), rolloff_max[0], label='Roll-off frequency (0.50)',color='b')
    ax[1].plot(librosa.times_like(rolloff_min), rolloff_min[0], label='Roll-off frequency (0.01)',color='w')

    ax[1].legend(loc='lower right')
    ax[1].set(title='log Power spectrogram')

    plt.savefig(f'{ft_jpg_output}{songname}_spec_rolloff.jpg')
    print("spec_rolloff complete.")

    ## poly_features 聚合特徵
    S = np.abs(librosa.stft(y))
    p0 = librosa.feature.poly_features(S=S)
    p1 = librosa.feature.poly_features(S=S, order=1)
    p2 = librosa.feature.poly_features(S=S, order=2)

    fig, ax = plt.subplots(nrows=4, sharex=True, figsize=(10, 8))
    times = librosa.times_like(p0)
    ax[0].plot(times, p0[0], label='order=0', alpha=0.8)
    # ax[0].plot(times, p1[1], label='order=1', alpha=0.8)
    # ax[0].plot(times, p2[2], label='order=2', alpha=0.8)
    ax[0].legend()
    ax[0].label_outer()
    ax[0].set(ylabel='Constant term ')
    # ax[1].plot(times, p1[0], label='order=1', alpha=0.8)
    ax[1].plot(times, p2[1], label='order=2', alpha=0.8, color='b')
    ax[1].set(ylabel='Linear term')
    ax[1].label_outer()
    ax[1].legend()
    ax[2].plot(times, p2[0], label='order=2', alpha=0.8, color='c')
    ax[2].set(ylabel='Quadratic term')
    ax[2].legend()
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                            y_axis='log', x_axis='time', ax=ax[3])


    plt.savefig(f'{ft_jpg_output}{songname}_poly_feature.jpg')
    print("poly_feature complete.")
    print("all complete!")

songname = sys.argv[1]
print_ft_jpg(songname)