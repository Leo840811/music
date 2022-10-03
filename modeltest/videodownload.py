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
from pytube import YouTube

def download(videourl,file_name):
    yt = YouTube(videourl)
    progMP4 = yt.streams.filter(progressive=True, file_extension='mp4')   # 設定下載方式及格式
    targetMP4 = progMP4.order_by('resolution').desc().first()  # 由畫質高排到低選畫質最高的來下載
    name = '%s.mp4' % (file_name)  # 給下載影片名
    video_file = targetMP4.download(output_path='./musicfile/mp4/%s' %(file_name) , filename=name) # 下載影片並給輸出路徑    
    


videourl = sys.argv[1]
file_name = sys.argv[2]
download(videourl,file_name)