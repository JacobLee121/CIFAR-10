#!/usr/bin/env python
#encoding:utf-8
"""
@author:
@time:2017/3/18 14:33
"""
# Import the functions to calculate feature descriptions
from skimage.feature import hog
import numpy as np
from sklearn.externals import joblib
from sklearn.svm import LinearSVC

# To read image file and save image feature descriptions
import os
import time
import glob
import pickle
import matplotlib.pyplot as plt
import operator
from sklearn import svm

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        DataDict = pickle.load(fo, encoding='bytes')
    return DataDict


def load_CIFAR_batch(filename):
    """load single batch of cifar"""
    with open(filename,"rb") as f:
        datadict = pickle.load(f,encoding='latin1')
        X = datadict['data']
        Y = datadict['labels']
        Y = np.array(Y)#字典载入的Y是list类型，把它转变成array类型
        return X, Y
def load_CIFAR_Labels(filename):
    with open(filename,'rb') as f:
        label_names = pickle.load(f,encoding='latin1')
        names = label_names['label_names']
        return names


def getFeat(TrainDataX, TestDataX):

    grayTempTest = rgb2gray(np.reshape(TestDataX[0,:],(32,32,3),order = 'F'))
    testFeature = hog(grayTempTest, orientations=9, pixels_per_cell=(8,8), cells_per_block=(4, 4), visualise=False, normalise=None)*1000
    print("testFeature", testFeature)

    grayTempTrain = rgb2gray(np.reshape(TrainDataX[0,:],(32,32,3),order = 'F'))
    trainFeature= hog(grayTempTrain, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(4, 4), visualise=False, normalise=None)*1000
    print("trainFeature",trainFeature)
    for data in TestDataX[1:,:]:
        image = np.reshape(data, (32, 32, 3),order = "F")
        gray = rgb2gray(image)

        fd = hog(gray, orientations=9, pixels_per_cell=(8,8), cells_per_block=(4, 4),visualise=False, normalise=None)*1000

        testFeature = np.vstack((testFeature,fd))

    path = os.getcwd()
    fd_path = os.path.join(path,'data','features','test', 'testFeature.pkl')
    joblib.dump(testFeature, fd_path,compress=3)
    print ("Test features are extracted and saved.")

    for data in TrainDataX[1:,:]:
        image = np.reshape(data, (32, 32, 3),order = "F")
        gray = rgb2gray(image)

        fd = hog(gray, orientations=9, pixels_per_cell=(8,8), cells_per_block=(4, 4), visualise=False, normalise=None)*1000

        trainFeature = np.vstack((trainFeature,fd))

    fd_path = os.path.join(path,'data','features','train', 'trainFeature.pkl')
    joblib.dump(trainFeature, fd_path,compress=3)
    print ("Train features are extracted and saved.")
    return trainFeature,testFeature
def rgb2gray(im):
    gray = np.dot(im[...,:3], [0.299, 0.587, 0.114])
    return gray
if __name__ == '__main__':
    t0 = time.time()
    filePath = r'E:\8MachineLearningProject\HOG-SVM-classifer-master Face\cifar-10-python\cifar-10-batches-py'
    # TrainData, TestData = getData(filePath)
    label_names = load_CIFAR_Labels("E:/8MachineLearningProject/HOG-SVM-classifer-master Face/cifar-10-python/cifar-10-batches-py/batches.meta")
    print(label_names)
    # MM,nn = load_CIFAR_batch("E:/8MachineLearningProject/HOG-SVM-classifer-master Face/cifar-10-python/cifar-10-batches-py/data_batch_1")
    # fdTest1 = np.reshape(MM[1,:], (32,32,3),order ="F")#需要按照Fortran形式的风格组合

    imgX1, imgY1 = load_CIFAR_batch("E:/8MachineLearningProject/HOG-SVM-classifer-master Face/cifar-10-python/cifar-10-batches-py/data_batch_1")
    imgX2, imgY2 = load_CIFAR_batch("E:/8MachineLearningProject/HOG-SVM-classifer-master Face/cifar-10-python/cifar-10-batches-py/data_batch_2")
    imgX3, imgY3 = load_CIFAR_batch("E:/8MachineLearningProject/HOG-SVM-classifer-master Face/cifar-10-python/cifar-10-batches-py/data_batch_3")
    imgX4, imgY4 = load_CIFAR_batch("E:/8MachineLearningProject/HOG-SVM-classifer-master Face/cifar-10-python/cifar-10-batches-py/data_batch_4")
    imgX5, imgY5 = load_CIFAR_batch("E:/8MachineLearningProject/HOG-SVM-classifer-master Face/cifar-10-python/cifar-10-batches-py/data_batch_5")
    Xte_rows, Yte = load_CIFAR_batch("E:/8MachineLearningProject/HOG-SVM-classifer-master Face/cifar-10-python/cifar-10-batches-py/test_batch")
    Xtr_rows = np.concatenate((imgX1, imgX2, imgX3, imgX4, imgX5))  # 每个图片占一列
    Ytr_rows = np.concatenate((imgY1, imgY2, imgY3, imgY4, imgY5))  # 每个图片的label  （0-9中的一个）占一列
    FeatureTrain, FeatureTestX = getFeat(Xtr_rows[:500], Xte_rows[:100])
    print('FeatureTrain',FeatureTrain[:10])
    print('FeatureTestX=',FeatureTestX)
    t1 = time.time()
    print('The cast of get feature time is:%f' % (t1 - t0))
    # print('fdTest[:10]',fdTest[:10],np.shape(fdTest),len(fdTest))
    # print('fdTrain[:10]',fdTrain[:10],np.shape(fdTrain))
    # # print ("Features are extracted and saved.")
    # # print("Features are extracted.")
    # clf_linear = LinearSVC()
    t0 = time.time()
    clf_rbf = svm.SVC(kernel='rbf',decision_function_shape='ovo',gamma=0.16).fit(FeatureTrain, Ytr_rows[:500])
    clf_linear = svm.SVC(kernel='linear').fit(FeatureTrain, Ytr_rows[:500])
    checkResult = clf_linear.predict(FeatureTestX)
    print("checkResult", checkResult)
    print(np.mean(checkResult == Yte[:100]))
    t1 = time.time()
    print('The cast of  time to train clf_rbf modle is:%f' % (t1 - t0))
    # clf_sigmoid = svm.SVC(kernel='sigmoid',decision_function_shape='ovo').fit(FeatureTrain, Ytr_rows[:500])
    # t2 = time.time()
    # print('The cast of  time to train clf_sigmoid modle is:%f' % (t2 - t1))
    # clf_poly = svm.SVC(kernel= 'poly', degree = 3,decision_function_shape='ovo').fit(FeatureTrain,Ytr_rows[:500])
    # t3 = time.time()
    # print('The cast of  time to train clf_poly modle is:%f' % (t3 - t2))
    # clf_linear = LinearSVC().fit(FeatureTrain,Ytr_rows[:500])
    # t3 = time.time()
    # print('The cast of  time to train clf_linear modle is:%f' % (t3 - t2))
    # print "Training a Linear SVM Classifier."

    # FeatureTrainX = joblib.load('E:\8MachineLearningProject\HOG-SVM-classifer-master Face\HOG-SVM-classifer-master\HOG+SVM classifer\HOG+SVM classifer\data\features\train\TrainFeature.pkl')
    # FeatureTestX = joblib.load('E:\8MachineLearningProject\HOG-SVM-classifer-master Face\HOG-SVM-classifer-master\HOG+SVM classifer\HOG+SVM classifer\data\features\test\TestFeature.pkl')
    # clf = LinearSVC()
    # clf.fit(FeatureTrain[:,100],Ytr_rows[:100])
    #print "Training a Linear SVM Classifier."
    #print(clf.fit(fdTrain,Ytr_rows[:200]))
    print("Modle is trained.")
    # for data in FeatureTestX[:100]:
    #     print(data,data.shape,clf_rbf.predict(data))
    reuslt = clf_rbf.predict(FeatureTestX[:100])
    print("reuslt_rbf",reuslt)
    print(clf_rbf.support_vectors_)
    print(clf_rbf.support_)
    print(clf_rbf.n_support_)
    print("Yte[:100]",Yte[:100])
    acc = np.mean(reuslt == Yte[:100])
    print(acc)
    # for i, clf in enumerate((clf_poly,clf_linear,clf_rbf,clf_sigmoid)):
    #     t0 = time.time()
    #     result = []
    #     # for data in FeatureTestX[:10]:
    #     #     result = np.vstack((result,clf.predict(data)))
    #     result = clf.predict(FeatureTrain[:100])
    #     print('FeatureTestX[:10]',FeatureTestX[:100])
    #     print('###############################################')
    #     print(clf)
    #     t1 = time.time()
    #     print('The cast of predict time is:%f' % (t1 - t0))
    #     print('result',result,type(result),len(result))
    #     # print('Yte',Yte,type(Yte),len(Yte))
    #     acc = np.mean(result == Ytr_rows[:100])
    #     print('acc=',acc,'%%%%%%%%%%%%%%%%%%%%%%%%')




