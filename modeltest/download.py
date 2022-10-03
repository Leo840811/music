from pytube import Playlist
from moviepy.editor import *
from pydub import AudioSegment
from pydub.utils import make_chunks
from pydub import AudioSegment
import os, re
import random
import math
import pandas as pd
import sys

def download(playlisturl,file_name):
    p = Playlist(playlisturl)
    n=0
    # 逐一處理撥放清單中的影片
    for video in p.videos:
        try:
            progMP4 = video.streams.filter(progressive=True, file_extension='mp4')   # 設定下載方式及格式
            targetMP4 = progMP4.order_by('resolution').desc().first()  # 由畫質高排到低選畫質最高的來下載
            n = n+1 #下一部影片
            name = '%s%s.mp4' % (file_name,n)  # 給下載影片名
            video_file = targetMP4.download(output_path='./musicfile/mp4/%s' %(file_name) , filename=name) # 下載影片並給輸出路徑
        except: #有錯誤時跳過
            print('%sError' %(n))
            pass

playlisturl = sys.argv[1]
file_name = sys.argv[2]
download(playlisturl,file_name)