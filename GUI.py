import threading
import tkinter as tk
import numpy as np
import sys
import simpleaudio
import pyAudioKits.audio as ak
import pyAudioKits.algorithm as alg
import tkinter.font as tkFont

sys.path.append(R'C:\Users\happywinder\Desktop\MASR-develop')
from masr.predict import MASRPredictor
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.font_manager as fm
from tkinter.filedialog import askopenfilename
import librosa
from work2 import detect as de
from train import predict
from work2.augment import awgn, spectral_subtraction, SNR_Calc

font_path = r'C:\Windows\Fonts\simhei.ttf'
# 加载字体文件
prop = fm.FontProperties(fname=font_path)


class MyWindow:
    def __init__(self, root):
        self.root = root
        self.root.geometry('1000x550')
        self.fontstyle = tkFont.Font(family="Lucida Grande", size=20)
        self.label = tk.Label(text='语音识别系统', font=self.fontstyle)
        self.label.place(x=400, y=5)
        self.button1 = tk.Button(text='特征提取', command=self.feature, width=10)
        self.button2 = tk.Button(text='端点检测', command=self.detect, width=10)
        self.button3 = tk.Button(text='说话人识别', command=self.classify, width=10)
        self.button4 = tk.Button(text='添加噪声', command=self.noise, width=10)
        self.button5 = tk.Button(text='语音增强', command=self.augment, width=10)
        self.button6 = tk.Button(text='语音识别', command=self.predict_, width=10)
        self.button7 = tk.Button(text='打开文件', command=self.openfile, width=10)
        self.button8 = tk.Button(text='播放音频', command=self.play, width=10)
        self.canvas = tk.Frame(bg='white', height=400, width=800)

        self.button7.place(x=50, y=50)
        self.button4.place(x=50, y=100)
        self.button5.place(x=50, y=150)
        self.button1.place(x=50, y=200)
        self.button2.place(x=50, y=250)
        self.button3.place(x=50, y=300)
        self.button6.place(x=50, y=350)
        self.button8.place(x=50, y=400)
        self.canvas.place(x=150, y=50)

    def play(self):
        p1 = threading.Thread(target=self.play_data_from_librosa, args=(self.signal, self.sr))
        p1.start()

    def openfile(self):
        self.wav_path = askopenfilename(filetypes=[("音频文件", "*.wav"), ("音频文件", "*.mp3")],
                                        initialdir=r"C:\Users\happywinder\Desktop\MASR-develop\dataset\audio\ST-CMDS-20170001_1-OS")
        if self.wav_path == '':
            return
        self.signal, self.sr = librosa.load(self.wav_path, sr=16000)
        fig = plt.figure(figsize=(8, 4))
        time = np.arange(0, len(self.signal)) / self.sr
        plt.plot(time, self.signal)
        plt.title('Waveform')
        plt.xlabel('time(s)')  # 采样率
        plt.ylabel('Amplitude')
        canvas = FigureCanvasTkAgg(fig, self.root)
        canvas.draw()
        canvas.get_tk_widget().place(x=150, y=50)

    def add_noise(self, data, coef=0.02):
        wn = np.random.normal(0, 1, len(data))
        # data_noise = np.where (data != 0.0，data. astype('float64' ) + coefon,0.0).astype (np.float32)
        data_noise = (data.astype('float64') + coef * wn).astype(np.float32)
        return data_noise

    def play_data_from_librosa(self, y, sr):
        player = simpleaudio.play_buffer(
            y,
            num_channels=1,
            bytes_per_sample=4,
            sample_rate=sr
        )
        try:
            player.wait_done()
        except KeyboardInterrupt:
            player.stop()

    def noise(self):
        self.signal = self.add_noise(self.signal, 0.01)
        fig = plt.figure(figsize=(8, 4))
        time = np.arange(0, len(self.signal)) / self.sr
        plt.plot(time, self.signal)
        plt.title('Waveform')
        plt.xlabel('time(s)')  # 采样率
        plt.ylabel('Amplitude')
        p1 = threading.Thread(target=self.play_data_from_librosa, args=(self.signal, self.sr,))
        p1.start()
        canvas = FigureCanvasTkAgg(fig, self.root)
        canvas.draw()
        canvas.get_tk_widget().place(x=150, y=50)

    def feature(self):
        spec = librosa.feature.melspectrogram(
            y=self.signal,  #
            sr=18000,  # 采样率
            n_mels=90,
            n_fft=900,  # FFT窗口的长度
            hop_length=200  # 步长
        )
        fig = plt.figure(figsize=(8, 4))
        plt.imshow(np.log2(spec))
        plt.xlabel('frequency/Hz')
        plt.ylabel('Amplitude/db')
        canvas = FigureCanvasTkAgg(fig, self.root)
        canvas.draw()
        canvas.get_tk_widget().place(x=150, y=50)

    def classify(self):
        label, ans = predict(self.wav_path, self.signal)
        self.label = tk.Label(text='说话人:' + label + '\n'
                                                    '预测为:' + ans, width=20)
        self.label.place(x=300, y=500)

    def detect(self):
        data = self.signal
        sr = self.sr
        fig = plt.figure(figsize=(8, 4))
        data = data.astype(np.float64)
        data /= np.max(data)
        N = len(data)
        wlen = 200
        inc = 500
        IS = 0.1
        NIS = int((IS * sr - wlen) // inc + 1)
        fn = (N - wlen) // inc + 1

        frameTime = de.calculate_frame_center_times(fn, wlen, inc, sr)
        time = [i / sr for i in range(N)]

        voiceseg, vsl, SF, NF, amp, zcr = de.voice_activity_detection_thresholding(data, wlen, inc, NIS)

        plt.subplot(3, 1, 1)
        plt.plot(time, data)
        plt.ylabel('波形图', fontdict=dict(fontfamily=prop.get_name()))
        plt.xlabel('时间', fontdict=dict(fontfamily=prop.get_name()))
        plt.subplot(3, 1, 2)
        plt.plot(frameTime, amp)
        plt.ylabel('短时能量', fontdict=dict(fontfamily=prop.get_name()))
        plt.xlabel('时间', fontdict=dict(fontfamily=prop.get_name()))
        plt.subplot(3, 1, 3)
        plt.plot(frameTime, zcr)
        plt.ylabel('过零率', fontdict=dict(fontfamily=prop.get_name()))
        plt.xlabel('时间', fontdict=dict(fontfamily=prop.get_name()))
        for i in range(vsl):
            plt.subplot(3, 1, 1)
            plt.axvline(x=frameTime[voiceseg[i]['start']], color='red', linestyle='--')
            plt.axvline(x=frameTime[voiceseg[i]['end']], color='red', linestyle='-')
            plt.subplot(3, 1, 2)
            plt.axvline(x=frameTime[voiceseg[i]['start']], color='red', linestyle='--')
            plt.axvline(x=frameTime[voiceseg[i]['end']], color='red', linestyle='-')
            plt.subplot(3, 1, 3)
            plt.axvline(x=frameTime[voiceseg[i]['start']], color='red', linestyle='--')
            plt.axvline(x=frameTime[voiceseg[i]['end']], color='red', linestyle='-')
        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, self.root)
        canvas.draw()
        canvas.get_tk_widget().place(x=150, y=50)

    def augment(self):
        f = ak.audio.Audio(self.signal, self.sr)
        f1 = alg.specSubstract(f, f[f.getDuration() - 0.3:])
        fig = plt.figure(figsize=(8, 4))
        f1.plot()
        self.signal = f1.samples
        p1 = threading.Thread(target=self.play_data_from_librosa, args=(self.signal, self.sr,))
        p1.start()
        canvas = FigureCanvasTkAgg(fig, self.root)
        canvas.draw()
        canvas.get_tk_widget().place(x=150, y=50)

    def predict_(self):

        predictor = MASRPredictor(model_tag='conformer_streaming_fbank_aishell')
        wav_path = self.wav_path
        result = predictor.predict(audio_data=self.signal, use_pun=False)
        score, text = result['score'], result['text']
        with open(self.wav_path[:-3] + 'txt', 'r', encoding='utf-8')as f:
            label = f.readline()
        self.label = tk.Label(text=f"识别结果: {text}\n"
                                   f"正确结果为:{label}", width=50 )
        self.label.place(x=500, y=500)


if __name__ == '__main__':
    root = tk.Tk()
    my_window = MyWindow(root)
    root.mainloop()
