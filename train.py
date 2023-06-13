import pickle

import librosa
from sklearn.mixture import GaussianMixture
import joblib
import os
import numpy as np
import math
from FeatureExtracter import extract_features
from tqdm import tqdm


def getGMM(filename):
    data = np.load(filename)
    print(data.shape)
    gmm = GaussianMixture(16, covariance_type='diag', random_state=0).fit(data)
    return gmm


def softmax(scores):
    ss = 0.0
    Sum = 0.0
    for score in scores:
        ss += score
    scores = [(-1) * float(i) / ss for i in scores]
    for score in scores:
        Sum += math.exp(score)
    # for score in scores:        
    # print("probalitiy:{0}, index:{1}".format(math.exp(max(scores)) / Sum, scores.index(max(scores))))
    return scores.index(max(scores))


def train_gmm_model():
    root_dir = './Feature_MFCC/'
    for item in tqdm(os.listdir(root_dir)):
        spk_mfcc = root_dir + item

        model_name = item[:-8] + '_MFCC_12_GMM.model'
        # print(model_file)
        gmm = getGMM(spk_mfcc)
        joblib.dump(gmm, 'model/' + model_name)


def test_gmm():
    test_data_list = []  # 将提取好的特征全部放入列表
    gmm_model_list = []  # 将加载好的模型全部放入gmm_model_list
    label = [0]
    k = 0
    test_file = os.listdir('test_MFCC')
    for item in os.listdir('model'):
        gmm_model = joblib.load('model/' + item)
        gmm_model_list.append(gmm_model)
    for item in test_file:
        data = np.load('test_MFCC/' + item)
        test_data_list.append(data)
    for i in range(1, len(test_file)):
        if test_file[i][:-13] == test_file[i - 1][:-13]:
            label.append(k)
        else:
            k += 1
            label.append(k)
    test_right = 0
    i = 0
    for test_data in tqdm(test_data_list):
        scores = []
        for gmm_model in gmm_model_list:
            test_score = gmm_model.score(test_data)
            scores.append(test_score)
        result = softmax(scores)
        print(f'label为{label[i]},预测为{result}')
        if label[i] == result:
            test_right += 1
        i += 1
        # print("-------------------")

    print("right:{0}, accuracy:{1}".format(test_right, test_right / 12000))


def predict(path, signal):
    gmm_model_list = []
    scores = []
    data = extract_features(signal, 16000, 0.025, 0.01)
    for item in os.listdir('model'):
        gmm_model = joblib.load('model/' + item)
        gmm_model_list.append(gmm_model)
    for gmm_model in gmm_model_list:
        test_score = gmm_model.score(data)
        scores.append(test_score)
    result = softmax(scores)
    num = result // 2 + 1
    alpha = 'A' if result % 2 == 0 else 'I'
    label = path[-12:-8]
    return label, '0' * (3 - len(str(num))) + str(num) + alpha


if __name__ == '__main__':
    train_gmm_model()
# os.mkdir('model')
# test_gmm()
# label, answer = predict('./test_MFCC/20170001P00002I0094_MFCC.npy')
# print(label, answer)
