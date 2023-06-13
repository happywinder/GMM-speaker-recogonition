import os

import numpy as np
import librosa
from txtFileReader import file_reader
from tqdm import tqdm
import matplotlib.pyplot as plt
from pyAudioAnalysis.audioBasicIO import read_audio_file
from pyAudioAnalysis.audioSegmentation import silence_removal


def trim_audio():
    path = '../../data/ST-CMDS-20170001_1-OS/'
    audio_file_list, label = file_reader('train_audio_speech.txt')
    result = np.array([])
    speaker = label[0]
    for i in tqdm(range(len(label))):
        if label[i] == speaker:
            signal, sr = librosa.load(path + audio_file_list[i], sr=16000)
            trimed_signal, _ = librosa.effects.trim(signal, top_db=30)
            # plot_signal(trimed_signal)
            result = np.concatenate((result, trimed_signal), axis=0)
            # plot_signal(result)
        elif label[i] != speaker:
            np.save('LumpAudio\\' + speaker + '.npy', result)
            result = np.array([])
            speaker = label[i]
    np.save('LumpAudio\\' + speaker + '.npy', result)


def plot_signal(signal):
    librosa.display.waveshow(signal)
    plt.show()


def mytrim(path):
    [Fs, x] = read_audio_file(path)
    segments = silence_removal(x, Fs, 0.020, 0.020, smooth_window=0.3, weight=0.3, plot=True)


if __name__ == '__main__':
    # os.mkdir('LumpAudio')
    mytrim(r"C:\Users\happywinder\Desktop\MASR-develop\dataset\audio\ST-CMDS-20170001_1-OS\20170001P00001A0001.wav")
# trim_audio()
