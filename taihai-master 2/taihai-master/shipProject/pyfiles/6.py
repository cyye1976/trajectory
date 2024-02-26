import numpy as np
np.random.seed(1337)
from keras.models import Model
from keras.layers import  Input,Conv1D,MaxPooling1D,UpSampling1D
import matplotlib.pyplot as plt
from keras.optimizers import Adam
import logging
from sklearn.metrics import recall_score, f1_score,precision_score
from KuhnMunkres import err_rate,best_map
from sklearn import mixture
from sklearn.metrics import classification_report
import argparse
def GMMcluster(data,y_true):
    #print('---------------jiangsu_GMM----------------')
    clf = mixture.GaussianMixture(n_components=2,covariance_type='full',random_state=2019)
    clf.fit(data)
    y_pred = clf.predict(data)
    rate, y_pred = err_rate(y_true, y_pred)
    print('\nAprecisionA{} %'.format(precision_score(y_true, y_pred, average='macro') * 100))
    print('\nAaccA{} %'.format((1 - rate) * 100))
    print('\nArecall_score_weightedA{} %'.format(recall_score(y_true, y_pred, average='macro') * 100))
    print('\nAf1_score_weightedA{} %'.format(f1_score(y_true, y_pred, average='macro') * 100))
    #target_names = ['class 0', 'class 1']
    #print("\n<classification_report:>{}".format(classification_report(y_true, y_pred, target_names=target_names)))
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="path")
    parser.add_argument('-i', '--inputPath',default='./')
    parser.add_argument('-o', '--outputPath',default='./')
    args = parser.parse_args()
    inputPath, inputPath2 = args.inputPath.split(',')
    y_true = np.load(inputPath2+'encoded_data_label.npy')
    X=np.load(inputPath+'encoded_data.npy')
    # print(X)
    GMMcluster(X,y_true)


