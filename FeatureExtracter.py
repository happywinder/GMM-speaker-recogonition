import numpy as np
import python_speech_features
import config
import numpy
import os
import librosa
from tqdm import tqdm
from txtFileReader import file_reader


# 从样本数组中提取梅尔频率倒谱系数
def extract_mfccs(sample_array, sampling_rate, winlen, winstep):
    """
    :param sample_array: signal 用于计算梅尔倒谱系数的音频信号
    :param sampling_rate: 音频采样率
    :param winlen:窗口长度 按秒计，默认0.025s(25ms)
    :param winstep:窗口步长 按秒计，默认0.01s（10ms）
    :return:回一个大小(numframes窗口数量 , numcep倒谱数量)的numpy数组
    """
    # numcep - 倒频谱返回的数量，默认13

    mfccs = python_speech_features.mfcc(sample_array, sampling_rate, winlen=winlen, winstep=winstep,
                                        numcep=13, winfunc=config.Windowing.hamming)
    mfccs = numpy.array(mfccs)
    # print(mfccs)
    mfccs = mfccs[:, :]
    return mfccs


# 此方法用于提取测试音频文件的mfccs
def extract_features(sample_array, sampling_rate, flen, hlen):
    mfccs = extract_mfccs(sample_array, sampling_rate, flen, hlen)
    return mfccs


if __name__ == '__main__':
    # for item in tqdm(os.listdir('LumpAudio')):
    #     signal = np.load('LumpAudio/' + item)
    #     mfccs = extract_features(signal, 16000, 0.025, 0.01)
    #     np.save('Feature_MFCC/'+item[:-4] + '_mfcc.npy', mfccs)
    # test文件的mfcc提取
    audio_name, label = file_reader('test_audio_speech.txt')
    # os.mkdir('test_MFCC')
    for item in audio_name:
        signal, sr = librosa.load('../../data/ST-CMDS-20170001_1-OS/'+item, sr=16000)
        trimed_signal, _ = librosa.effects.trim(signal, top_db=30)
        mfccs = extract_features(trimed_signal, 16000, 0.025, 0.01)
        np.save('test_MFCC/' + item[:-4] + '_MFCC', mfccs)
        print(mfccs.shape)
        break
