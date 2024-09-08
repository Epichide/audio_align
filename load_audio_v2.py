#!/usr/bin/python3
# --*-- coding: utf-8 --*--
# @Author: EPICHIDE
# @Email: no email
# @Time: 2024/9/7 19:14
# @File: load_audio.py
# @Software: PyCharm
from moviepy.editor import AudioFileClip, CompositeAudioClip
import librosa
import numpy as np
import scipy.signal
from pydub import AudioSegment
from pydub.playback import play
import time
from moviepy.editor import concatenate_audioclips
from matplotlib import  pyplot as plt
from moviepy.editor import AudioFileClip,VideoFileClip,AudioClip
import os
def get_basename(filepath):
    basename=os.path.basename(filepath)
    basename=basename.split(".")[0]
    return basename

# ================== load audio ====================
def load_by_audiosegment_play(vediofile):
    t1=time.time()
    #vediofile="1.mp4"
    audiofile=get_basename(vediofile)+".wav"
    audiofile="temp.wav"
    my_vedio_clip = VideoFileClip(vediofile)
    my_audio_clip=my_vedio_clip.audio
    my_audio_clip.write_audiofile(audiofile,fps=8000)

    audio_seg=AudioSegment.from_wav(audiofile)
    sample_rate=audio_seg.frame_rate
    sample_width=audio_seg.sample_width
    num_channels=audio_seg.channels
    num_samples=audio_seg.frame_count()
    audio_rawdata=audio_seg.raw_data
    duration=audio_seg.duration_seconds
    print("sample rate:",sample_rate,"HZ")
    print("num channels:",num_channels)
    print("num samples:",num_samples)
    print("duaration:",duration,"second")

    # play audio
    #play(audio_seg)

    # plot mel
    audio_array=np.array(audio_seg.get_array_of_samples()) # int16
    audio_array_float=np.array(audio_array,dtype=np.float32)
    N_FFT = 512
    N_MELS = 80
    N_MFCC = 13

    mel_spec = librosa.feature.melspectrogram(y=audio_array_float,
                                              sr=sample_rate,
                                              n_fft=N_FFT,
                                              hop_length=sample_rate // 100,
                                              win_length=sample_rate // 40,
                                              n_mels=N_MELS)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel_spec), n_mfcc=N_MFCC)
    librosa.display.specshow(data=mfcc,
                             sr=sample_rate,
                             n_fft=N_FFT,
                             hop_length=sample_rate // 100,
                             win_length=sample_rate // 40,
                             x_axis="s")
    #plt.show()

def load_by_audiosegment(vediofile):
    t1=time.time()
    #vediofile="1.mp4"
    audiofile=get_basename(vediofile)+".wav"
    my_vedio_clip = VideoFileClip(vediofile)
    my_audio_clip=my_vedio_clip.audio
    my_audio_clip.write_audiofile(audiofile,fps=8000,ffmpeg_params=["-ac","1"])

    audio_seg=AudioSegment.from_wav(audiofile)
    sample_rate=audio_seg.frame_rate
    sample_width=audio_seg.sample_width
    num_channels=audio_seg.channels
    num_samples=audio_seg.frame_count()
    audio_rawdata=audio_seg.raw_data
    duration=audio_seg.duration_seconds
    print("sample rate:",sample_rate,"HZ")
    print("num channels:",num_channels)
    print("num samples:",num_samples)
    print("duaration:",duration,"second")

    # play audio
    #play(audio_seg)

    # plot mel
    audio_array=np.array(audio_seg.get_array_of_samples()) # int16
    audio_array_float=np.array(audio_array,dtype=np.float32)
    print(audio_array_float.shape)
    return audio_array_float,audio_seg

def load_audio(vediofile,sample_rate=8000,nbytes=2):
    audio_clip=AudioFileClip(vediofile,fps=sample_rate,nbytes=nbytes)
    sample_rate=audio_clip.fps
    sample_width= nbytes
    num_channels=audio_clip.nchannels
    duration=audio_clip.duration
    audio_samples=audio_clip.to_soundarray(fps=sample_rate)[:,0] # mono channel
    audio_clip.close()
    # == show audio info
    print("sample rate:",sample_rate,"HZ")
    print("num channels",num_channels)
    print("num samples:",len(audio_samples))
    print("duration:",duration,"second")
    return audio_samples

# ================ mix_audio ===============


def mix_audio(signal1,signal2,shift=0):
    if shift<0:
        signal2=signal2[shift:]
    else:
        signal1=signal1[shift:]
    maxlen=max(len(signal2),len(signal1))
    if len(signal1)<maxlen:
        signal1=np.pad(signal1,(0,maxlen-len(signal1)))
    if len(signal2)<maxlen:
        signal2=np.pad(signal2,((0,maxlen-len(signal2))))
    signal=signal1+signal2
    signal=np.array(signal,dtype=np.int16)
    merge_seg=AudioSegment(signal.tobytes(),
                           frame_rate=8000,
                           sample_width=2,
                           channels=1)
    play(merge_seg)


def mix_audio_moviepy(videos,sample_rate=8000,shift_seconds=[]):
    audios=[AudioFileClip(video,fps=sample_rate) for video in videos]
    audio_crops=[audio.subclip(second,None) for second,audio in zip(shift_seconds,audios)]

    audio_combined = CompositeAudioClip(audio_crops)  # 融合音频
    audio_combined.fps=8000
    audio_combined.write_audiofile('combined_audio.wav')  # 输出融合后的音频文件
    for audio_crop in audio_crops: audio_crop.close()
    for audio in audios:audio.close()
    audio_combined.close()
# ================== align audio=================
def correlate(signal1,signal2,sample_rate,method="fft",absnorm=True):
    if absnorm:
        signal1=np.abs(signal1)
        signal2=np.abs(signal2)
        mean1=np.nanmean(signal1)
        mean2=np.nanmean(signal2)
        signal1=(signal1-mean1)/(np.nanmax(signal1)-mean1)
        signal2=(signal2-mean2)/(np.nanmax(signal2)-mean2)
    if method=="convolve":
        c21 = np.correlate(signal1, signal2, mode='full')
    elif method=="fft":
        c21 = scipy.signal.fftconvolve(signal1, signal2[::-1], mode='full')
    t21 = np.argmax(c21)
    lags = np.arange(-len(signal2) + 1, len(signal1))
    lag = lags[t21]
    fast_second = lags[t21] / sample_rate


    # plt.plot(signal1[lag:])
    # # plt.plot(signal1,color="#88001155")
    # plt.plot(signal2[:],color="#55555555")
    # plt.show()
    # mix_audio(signal1,signal2,shift=lags[t21])
    return lag, fast_second

def audio_align(video1,vedio2,sample_rate=8000):
    signal1=load_audio(video1,sample_rate=sample_rate)
    signal2= load_audio(vedio2,sample_rate=sample_rate)
    lag, fast_second=correlate(signal1,signal2,
                               sample_rate=sample_rate,
                               method="convolve",absnorm=True)
    # mix_audio(signal1,signal2,shift=0)
    print(video1,
          " is ",fast_second,
          " second faster than ",vedio2,",shift :",lag)
    # mix_audio(signal1,signal2,shift=lags[t21])
    return lag,fast_second

def audios_align(videos,sample_rate=8000):
    if len(videos)==1:return [0]
    signalss=[load_audio(video,sample_rate=sample_rate) for video in videos]
    ref_signals=signalss[0]
    lags=[0]
    delay_seconds=[0]
    for tar_signals in signalss[1:]:
        lag,fast_second=correlate(tar_signals,ref_signals,
                                  sample_rate=sample_rate,
                                  method="fft",
                                  absnorm=True)
        lags.append(lag)
        delay_seconds.append(fast_second)
    delays=np.array(delay_seconds)
    lags=np.array(lags,dtype=np.int32)
    delays-=np.min(delays)
    lags-=np.min(lags)
    return lags,delays

import ffmpeg
def get_videos_infos(videos):
    fpss=[]
    num_framess=[]
    for video in videos:
        probe=ffmpeg.probe(video)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        num_frames = int(video_stream['nb_frames'])
        fps = int(video_stream['r_frame_rate'].split('/')[0]) / int(video_stream['r_frame_rate'].split('/')[1])
        duration = float(video_stream['duration'])
        fpss.append(fps)
        num_framess.append(num_frames)
    return fpss
def videos_align(videos):
    t1=time.time()
    FPS=44100
    delay_seconds=audios_align(videos,sample_rate=FPS)[1]
    t2=time.time()
    fpss=get_videos_infos(videos)

    delay_frames=[int(fps*delay_second) for fps,delay_second in zip(fpss,delay_seconds)]
    print("="*30)
    print("cost time:",t2-t1)
    print("delay seconds:",delay_seconds)
    print("delay frames:",delay_frames)
    print("="*30)

    mix_audio_moviepy(videos,sample_rate=FPS,shift_seconds=delay_seconds)

if __name__ == '__main__':
    videos_align([r"1.mp4",r"2.mp4","3.mp4"])
    videos_align([r"4.mp4",r"5.mp4"])
