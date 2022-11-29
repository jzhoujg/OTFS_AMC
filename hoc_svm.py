import scipy.io as scio
import torch
import numpy as np
import pylab as pl  # 绘图功能
from sklearn import svm
import joblib


training = False
data_otfs = scio.loadmat("./HOC/otfs_nm.mat")
data_ofdm = scio.loadmat("./HOC/ofdm_nm.mat")

otfs = np.array(data_otfs['HOC_L'])
ofdm = np.array(data_ofdm['HOC_L'])
otfs = otfs.T
ofdm = ofdm.T


test_data_otfs = scio.loadmat("E:\Projects\OTFS_MODULATIONS_IDENTIFICATION\OTFS_SYN\HOC\otfs_nm_test_15.mat")
test_data_ofdm = scio.loadmat("E:\Projects\OTFS_MODULATIONS_IDENTIFICATION\OTFS_SYN\HOC\ofdm_nm_test_15.mat")

test_otfs = np.array(test_data_otfs['HOC_L'])
test_ofdm = np.array(test_data_ofdm['HOC_L'])
test_otfs = test_otfs.T
test_ofdm = test_ofdm.T


X = np.zeros([36000,6])
X[:18000,0:3] = otfs.real
X[:18000,3:6] = otfs.imag

X[18000:,0:3] = ofdm.real
X[18000:,3:6] = ofdm.imag

X_test = np.zeros([3600,6])
X_test[:1800,0:3] = test_otfs.real
X_test[:1800,3:6] = test_otfs.imag
X_test[1800:,0:3] = test_ofdm.real
X_test[1800:,3:6] = test_ofdm.imag

Y = [0] * 18000 + [1] * 18000

Y_test = [0] * 1800 + [1] * 1800
if training:
    clf = svm.SVC(kernel='rbf')
    clf.fit(X, Y)
    joblib.dump(clf, '''train_model.m''')
    print("Model is Ready!")
else:
    clf = joblib.load("train_model.m")
    print("Model loaded!")
    otfs_acc_num = 0
    ofdm_acc_num = 0

    print(clf.predict(X_test))

    for i in range(3600):
        if Y_test[i] == 0 and clf.predict([X_test[i]]) == 0:
            otfs_acc_num += 1
        if Y_test[i] == 1 and clf.predict([X_test[i]]) == 1:
            ofdm_acc_num += 1
    print("otfs_acc")
    print(otfs_acc_num/1800)

    print("ofdm_acc")
    print(ofdm_acc_num/1800)
    print("acc")
    print((otfs_acc_num+ofdm_acc_num)/3600)