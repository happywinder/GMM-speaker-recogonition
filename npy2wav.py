import numpy as np


def vec2wav(pcm_vec, wav_file, framerate=16000):
    """
    将numpy数组转为单通道wav文件
    :param pcm_vec: 输入的numpy向量
    :param wav_file: wav文件名
    :param framerate: 采样率
    :return:
    """
    import wave

    # pcm_vec = np.clip(pcm_vec, -32768, 32768)

    if np.max(np.abs(pcm_vec)) > 1.0:
        pcm_vec *= 32767 / max(0.01, np.max(np.abs(pcm_vec)))
    else:
        pcm_vec = pcm_vec * 32768
    pcm_vec = pcm_vec.astype(np.int16)
    wave_out = wave.open(wav_file, 'wb')
    wave_out.setnchannels(1)
    wave_out.setsampwidth(2)
    wave_out.setframerate(framerate)
    wave_out.writeframes(pcm_vec)


vec2wav(np.load('LumpAudio/20170001P00001A.npy'),'test.wav')
