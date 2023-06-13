import numpy


class Paths:
    DATA_PATH = '../../data/ST-CMDS-20170001_1-OS/'
    TEST_PATH = "测试文件/test3.WAV"


class Audio:
    # 采样率
    samp_rate = 16000
    # 采样点个数(音频处理单元，也就是一个音频帧里有多少次采样)
    fsize = 400
    # 每一音频帧的时间长度
    # 每秒有16000次采样，而每个帧有400次采样，所以一个帧的时间是400/160000
    flen = 0.025

    # 连续窗口之间的步长
    hlen = 0.01


class Windowing:
    hamming = lambda x: 0.54 - 0.46 * numpy.cos((2 * numpy.pi * x) / (400 - 1))


class Model:
    # KFold的n_splits
    n_splits = 1000
